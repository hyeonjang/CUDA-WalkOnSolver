#pragma once
#include <cuwos/common.h>
#include <cuwos/vector.h>


template <typename T>
void debug_memory(T* gpu_mem, size_t size) {
    std::vector<T> to_print(size);
    CHECK_CUDA(cudaMemcpy(to_print.data(), gpu_mem, sizeof(T)*size, cudaMemcpyDeviceToHost));
    fmt::print("{} \n", fmt::join(to_print, ","));
}

template <typename T, typename U, SparseFormat format>
void print_sparse(const SparseMatrixDynamic<T, U, format>& m) {
    debug_memory(m.rows(), m.nrow());
    debug_memory(m.ncol(), m.n_col_indices());
    debug_memory(m.values(), m.nnz());
} 