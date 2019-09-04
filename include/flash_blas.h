// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <functional>
#include "logger.h"
#include "pointers/allocator.h"
#include "pointers/pointer.h"
#include "types.h"

namespace flash {

  // C = alpha*A*B + beta*C
  FBLAS_INT gemm(CHAR mat_ord, CHAR trans_a, CHAR trans_b, FBLAS_UINT m,
                 FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha, FPTYPE beta,
                 flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c,
                 FBLAS_UINT lda_a = 0, FBLAS_UINT lda_b = 0,
                 FBLAS_UINT lda_c = 0);

  // y = alpha*A*x + beta*y
  FBLAS_INT gemv(CHAR mat_ord, CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n,
                 FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                 flash_ptr<FPTYPE> x, flash_ptr<FPTYPE> y);
}  // namespace flash
