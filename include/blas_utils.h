// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <parallel/algorithm>
#include "tasks/task.h"
#include "types.h"

namespace flash {
  struct SparseBlock {
    // Offsets (Row/Col)
    int *offs = nullptr;

    // Bon-zero indices (on flash)
    flash_ptr<int> idxs_fptr;
    int *          idxs_ptr = nullptr;

    // Non-zero vals (on flash)
    flash_ptr<FPTYPE> vals_fptr;
    FPTYPE *          vals_ptr = nullptr;

    // BLOCK DESCRIPTORS
    // Block start (Row/Col)
    int start;
    // Matrix Dims (CSR/CSC)
    int nrows;
    int ncols;
    // Block size (Row/Col)
    int blk_size;

    SparseBlock() {
      this->offs = nullptr;
      this->idxs_ptr = nullptr;
      this->vals_ptr = nullptr;
      this->start = 0;
      this->nrows = 0;
      this->ncols = 0;
      this->blk_size = 0;
    }

    SparseBlock(const SparseBlock &other) {
      this->offs = other.offs;
      this->idxs_fptr = other.idxs_fptr;
      this->vals_fptr = other.vals_fptr;
      this->idxs_ptr = other.idxs_ptr;
      this->vals_ptr = other.vals_ptr;
      this->start = other.start;
      this->nrows = other.nrows;
      this->ncols = other.ncols;
      this->blk_size = other.blk_size;
    }
  };

  // Given a <flash_ptr, in_mem_ptr> mapping, obtain indices
  // and values pointers for given SparseBlock
  inline void fill_sparse_block_ptrs(
      std::unordered_map<flash::flash_ptr<void>, void *, flash::FlashPtrHasher,
                         flash::FlashPtrEq> &in_mem_ptrs,
      SparseBlock &                          blk) {
    if (in_mem_ptrs.find(blk.idxs_fptr) == in_mem_ptrs.end()) {
      GLOG_FATAL("idxs fptr not found in in_mem_ptrs");
    }
    if (in_mem_ptrs.find(blk.vals_fptr) == in_mem_ptrs.end()) {
      GLOG_FATAL("vals fptr not found in in_mem_ptrs");
    }
    blk.idxs_ptr = (decltype(blk.idxs_ptr)) in_mem_ptrs[blk.idxs_fptr];
    blk.vals_ptr = (decltype(blk.vals_ptr)) in_mem_ptrs[blk.vals_fptr];
  }

  // for sparse matrices in CSR format only
  inline FBLAS_UINT get_next_blk_size(int *offs_ptr, int nrows, int min_size,
                                      int max_size) {
    FBLAS_UINT max_nnzs = MAX_NNZS;
    FBLAS_UINT blk_size = min_size;
    while (blk_size < (FBLAS_UINT) nrows &&
           ((FBLAS_UINT)(offs_ptr[blk_size] - offs_ptr[0]) <= max_nnzs)) {
      blk_size++;
    }

    return std::min(blk_size, (FBLAS_UINT) max_size);
  }

  inline void fill_blocks(int *offs, FBLAS_UINT n_rows,
                          std::vector<FBLAS_UINT> &blk_sizes,
                          std::vector<FBLAS_UINT> &offsets,
                          FBLAS_UINT min_blk_size, FBLAS_UINT max_blk_size) {
    FBLAS_UINT cur_start = 0;
    while (cur_start < n_rows) {
      FBLAS_UINT cblk_size = flash::get_next_blk_size(
          offs + cur_start, n_rows - cur_start, min_blk_size, max_blk_size);
      blk_sizes.push_back(cblk_size);
      offsets.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
  }
}  // namespace flash
