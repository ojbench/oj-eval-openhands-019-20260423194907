#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    
    // Move current query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    
    // Concatenate first i+1 keys vertically - do this in HBM first for efficiency
    Matrix* concatenated_keys = matrix_memory_allocator.Allocate("concatenated_keys");
    if (i == 0) {
      gpu_sim.Copy(keys[0], concatenated_keys, kInGpuHbm);
    } else {
      gpu_sim.Copy(keys[0], concatenated_keys, kInGpuHbm);
      for (size_t j = 1; j <= i; ++j) {
        Matrix* temp_key = matrix_memory_allocator.Allocate("temp_key");
        gpu_sim.Copy(keys[j], temp_key, kInGpuHbm);
        
        Matrix* new_concat = matrix_memory_allocator.Allocate("new_concat");
        gpu_sim.Concat(concatenated_keys, temp_key, new_concat, 0, kInGpuHbm);
        
        gpu_sim.ReleaseMatrix(concatenated_keys);
        gpu_sim.ReleaseMatrix(temp_key);
        concatenated_keys = new_concat;
      }
    }
    
    // Move concatenated keys to SRAM for transpose
    gpu_sim.MoveMatrixToSharedMem(concatenated_keys);
    
    // Transpose concatenated keys for matrix multiplication
    Matrix* keys_transposed = matrix_memory_allocator.Allocate("keys_transposed");
    gpu_sim.Copy(concatenated_keys, keys_transposed, kInSharedMemory);
    gpu_sim.Transpose(keys_transposed, kInSharedMemory);
    
    // Compute Q * K^T
    Matrix* qk_result = matrix_memory_allocator.Allocate("qk_result");
    gpu_sim.MatMul(current_query, keys_transposed, qk_result);
    
    // Apply softmax more efficiently - compute exp of entire matrix first
    Matrix* exp_result = matrix_memory_allocator.Allocate("exp_result");
    gpu_sim.MatExp(qk_result, exp_result);
    
    // For each row, compute sum and divide
    Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_result");
    
    for (size_t row = 0; row < exp_result->GetRowNum(); ++row) {
      // Get current row from exp_result
      Matrix* current_exp_row = matrix_memory_allocator.Allocate("current_exp_row");
      gpu_sim.GetRow(exp_result, row, current_exp_row, kInSharedMemory);
      
      // Sum all exp values in this row
      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp");
      gpu_sim.Sum(current_exp_row, sum_exp);
      
      // Divide each element by sum to get softmax
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(current_exp_row, sum_exp, softmax_row);
      
      if (row == 0) {
        gpu_sim.Copy(softmax_row, softmax_result, kInSharedMemory);
      } else {
        Matrix* temp_softmax = matrix_memory_allocator.Allocate("temp_softmax");
        gpu_sim.Copy(softmax_result, temp_softmax, kInSharedMemory);
        gpu_sim.Concat(temp_softmax, softmax_row, softmax_result, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(temp_softmax);
      }
      
      gpu_sim.ReleaseMatrix(current_exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);
      gpu_sim.ReleaseMatrix(softmax_row);
    }
    
    // Concatenate first i+1 values vertically - do this in HBM first
    Matrix* concatenated_values = matrix_memory_allocator.Allocate("concatenated_values");
    if (i == 0) {
      gpu_sim.Copy(values[0], concatenated_values, kInGpuHbm);
    } else {
      gpu_sim.Copy(values[0], concatenated_values, kInGpuHbm);
      for (size_t j = 1; j <= i; ++j) {
        Matrix* temp_value = matrix_memory_allocator.Allocate("temp_value");
        gpu_sim.Copy(values[j], temp_value, kInGpuHbm);
        
        Matrix* new_concat = matrix_memory_allocator.Allocate("new_concat");
        gpu_sim.Concat(concatenated_values, temp_value, new_concat, 0, kInGpuHbm);
        
        gpu_sim.ReleaseMatrix(concatenated_values);
        gpu_sim.ReleaseMatrix(temp_value);
        concatenated_values = new_concat;
      }
    }
    
    // Move concatenated values to SRAM for final multiplication
    gpu_sim.MoveMatrixToSharedMem(concatenated_values);
    
    // Compute final result: softmax_result * V
    Matrix* final_result = matrix_memory_allocator.Allocate("final_result");
    gpu_sim.MatMul(softmax_result, concatenated_values, final_result);
    
    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(final_result);
    
    // Run simulator and commit answer
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*final_result);
    
    // Clean up temporary matrices
    gpu_sim.ReleaseMatrix(concatenated_keys);
    gpu_sim.ReleaseMatrix(keys_transposed);
    gpu_sim.ReleaseMatrix(qk_result);
    gpu_sim.ReleaseMatrix(exp_result);
    gpu_sim.ReleaseMatrix(softmax_result);
    gpu_sim.ReleaseMatrix(concatenated_values);
    gpu_sim.ReleaseMatrix(final_result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu