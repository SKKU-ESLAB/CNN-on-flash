// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
//#include "mkl.h"
#include "pointers/pointer.h"
#include "tasks/task.h"
#include "types.h"
#include "utils.h"

#include <streamline_annotate.h>
#include <fstream>
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "cblas.h"

using namespace arm_compute;

namespace flash {
  // C = alpha*A*B + beta*C
  class GemmTask : public BaseTask {
    flash_ptr<FPTYPE>       matA, matB, matC;
    int                     a_nrows, a_ncols, b_ncols;
    int                     lda_a, lda_b, lda_c;
    FPTYPE                  alpha, beta;
    decltype(CblasNoTrans)  trans_a, trans_b;
    decltype(CblasRowMajor) mat_ord;

   public:
    GemmTask(flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c,
             FBLAS_UINT a_nrows, FBLAS_UINT a_ncols, FBLAS_UINT b_ncols,
             FBLAS_UINT ptr_offset[3], FBLAS_UINT lda_a, FBLAS_UINT lda_b,
             FBLAS_UINT lda_c, StrideInfo stride_info[3], FPTYPE alpha,
             FPTYPE beta, CHAR trans_a, CHAR trans_b, CHAR mat_ord) {
      this->alpha = alpha;
      this->beta = beta;
      this->trans_a = (trans_a == 'T' ? CblasTrans : CblasNoTrans);
      this->trans_b = (trans_b == 'T' ? CblasTrans : CblasNoTrans);
      this->mat_ord = (mat_ord == 'R' ? CblasRowMajor : CblasColMajor);
      this->matA = a + ptr_offset[0];
      this->matB = b + ptr_offset[1];
      this->matC = c + ptr_offset[2];

      // printf("ptr offests : %d %d %d\n", ptr_offset[0], ptr_offset[1],
      //        ptr_offset[2]);

      this->a_nrows = a_nrows;
      this->a_ncols = a_ncols;
      this->b_ncols = b_ncols;
      this->lda_a = lda_a;
      this->lda_b = lda_b;
      this->lda_c = lda_c;

      this->add_read(this->matA, stride_info[0]);
      this->add_read(this->matB, stride_info[1]);
      // if source is not required, don't read
      if (beta != 0.0f) {
        this->add_read(this->matC, stride_info[2]);
      } else {
        GLOG_INFO("not adding read");
      }

      this->add_write(this->matC, stride_info[2]);
    }

    void print_matrix(FPTYPE* a, FBLAS_UINT M, FBLAS_UINT N) {
      // printf("\nMatrix %s\n", s.c_str());
      for (FBLAS_UINT i = 0; i < M; i++) {
        for (FBLAS_UINT j = 0; j < N; j++)
          printf("%.1f ", a[i * N + j]);
        printf("\n");
      }
    }

    void execute() {
      // ANNOTATE_SETUP;
      static std::atomic<FBLAS_UINT> cnt(0);
      GLOG_DEBUG("Executing tsk#", cnt.fetch_add(1));
      // mkl_set_num_threads_local(GEMM_MKL_NTHREADS);
      FPTYPE* a_ptr = (FPTYPE*) in_mem_ptrs[matA];
      FPTYPE* b_ptr = (FPTYPE*) in_mem_ptrs[matB];
      FPTYPE* c_ptr = (FPTYPE*) in_mem_ptrs[matC];
      FPTYPE* tmp_ptr;
      if (beta != 0.)
        tmp_ptr = (FPTYPE*) malloc(b_ncols * a_nrows * sizeof(float));
      // tmp_ptr = (FPTYPE*) aligned_alloc(SECTOR_LEN,
      //                                  b_ncols * a_nrows * sizeof(float));
      GLOG_ASSERT(a_ptr != nullptr, "null a_ptr");
      GLOG_ASSERT(b_ptr != nullptr, "null b_ptr");
      GLOG_ASSERT(c_ptr != nullptr, "null c_ptr");
      GLOG_INFO("MKL params : trans_a:", trans_a == CblasTrans ? 'T' : 'N',
                ", trans_b:", trans_b == CblasTrans ? 'T' : 'N',
                ", a_nrows:", a_nrows, ", b_ncols:", b_ncols,
                ", a_ncols:", a_ncols, ", alpha:", alpha, ", beta:", beta,
                ", lda_a:", lda_a, ", lda_b:", lda_b, ", lda_c:", lda_c);

      /*
        for (int i = 0; i < a_nrows; i++) {
          for (int j = 0; j < a_nrows; j++) {
            printf("%d ", (int) a_ptr[i * a_nrows + j]);
          }
          printf("\n");
        }
        printf("\n");
        for (int i = 0; i < a_nrows; i++) {
          for (int j = 0; j < a_nrows; j++) {
            printf("%d ", (int) b_ptr[i * a_nrows + j]);
          }
          printf("\n");
        }
        printf("\n");
        for (int i = 0; i < a_nrows; i++) {
          for (int j = 0; j < a_nrows; j++) {
            printf("%d ", (int) c_ptr[i * a_nrows + j]);
          }
          printf("\n");
        }
        printf("\n");
      */
      Tensor     tensor_a, tensor_b, tensor_c, tensor_tmp;
      TensorInfo info_a, info_b, info_c, info_tmp;
      NEGEMM     gemm;

      // ANNOTATE("tensor init start");
      NEArithmeticAddition add;  // HY
      info_a.init(TensorShape(a_ncols, a_nrows), 1, DataType::F32);
      info_b.init(TensorShape(b_ncols, a_ncols), 1, DataType::F32);
      info_c.init(TensorShape(b_ncols, a_nrows), 1, DataType::F32);
      info_tmp = TensorInfo(TensorShape(b_ncols, a_nrows), 1, DataType::F32);

      tensor_a.allocator()->init(info_a);
      tensor_b.allocator()->init(info_b);
      tensor_c.allocator()->init(info_c);
      if (beta != 0.)
        tensor_tmp.allocator()->init(info_tmp);

      // ANNOTATE("gemm configure");
      // gemm.configure(&tensor_a, &tensor_b, &tensor_c, &tensor_tmp, alpha,
      // beta);
      if (beta == 0.) {
        gemm.configure(&tensor_a, &tensor_b, nullptr, &tensor_c, alpha, 0.);
      } else {
        gemm.configure(&tensor_a, &tensor_b, nullptr, &tensor_tmp, alpha, 0.);
        add.configure(&tensor_tmp, &tensor_c, &tensor_c, ConvertPolicy::WRAP);
      }
      // ANNOTATE("tensor allocate");
      tensor_a.allocator()->import_memory(a_ptr,
                                          a_ncols * a_nrows * sizeof(float));
      tensor_b.allocator()->import_memory(b_ptr,
                                          b_ncols * a_ncols * sizeof(float));
      tensor_c.allocator()->import_memory(c_ptr,
                                          b_ncols * a_nrows * sizeof(float));
      if (beta != 0.)
        tensor_tmp.allocator()->import_memory(
            tmp_ptr, b_ncols * a_nrows * sizeof(float));

      // ANNOTATE("gemm run start");
      if (beta == 0.) {
        gemm.run();
      } else {
        gemm.run();
        add.run();
      }
      // ANNOTATE("gemm run end");

      /* Init Window */
      Window c_window;
      c_window.use_tensor_dimensions(tensor_c.info()->tensor_shape(),
                                     Window::DimY);
      Iterator c_it(&tensor_c, c_window);
      // float*   tmp_ptr = (float*) tensor_tmp.buffer();
      // GLOG_INFO("[Address]tmp: ", tmp_ptr);
      /*
            for (int i = 0; i < a_nrows; i++) {
              for (int j = 0; j < a_nrows; j++) {
                printf("%d ", (int) tmp_ptr[i * a_nrows + j]);
              }
              printf("\n");
            }
            printf("\n");
      */

      /* Essential: Copy the result in tmp to C */
      // ANNOTATE("copy window start");
      /*
      execute_window_loop(c_window
                          [&](const Coordinates& id) {
                    memcpy(c_it.ptr(), tmp_ptr + id.y() * a_nrows,
                           a_nrows * sizeof(float));
                          },
                          c_it);
    */
      // ANNOTATE("copy window end");

      std::string outname;
      outname = "task_result_acl/" + std::to_string(a_nrows) + "_task" +
                std::to_string(this->task_id);  // task number
      outname.append(".txt");
      std::ofstream outfile(outname.c_str());

      for (int i = 0; i < a_nrows; i++) {
        for (int j = 0; j < a_nrows; j++) {
          // printf("%d ", (int) c_ptr[i * a_nrows + j]);
          outfile << (int) c_ptr[i * a_nrows + j] << " ";
        }
        // printf("\n");
        outfile << "\n";
      }
      // printf("\n");
      outfile.close();

      // printf("task[%d]\n", this->task_id);
      /*      printf("matA: ");
            for (int i = 0; i < 10; i++) {
              printf("%d ", (int) a_ptr[i]);
            }
            printf("\nmatB: ");
            for (int i = 0; i < 10; i++) {
              printf("%d ", (int) b_ptr[i]);
            }
      */

      /*
      printf("matC: ");
      for (int i = 0; i < a_nrows; i++) {
        for (int j = 0; j < a_nrows; j++) {
          printf("%d ", (int) c_ptr[i * a_nrows + j]);
        }
        printf("\n");
      }
      printf("\n");
      */

      /*
      printf("matC: ");
      for (int i = 0; i < 10; i++) {
        printf("%d ", (int) c_ptr[i]);
      }
      printf("\n");
      */

      printf("task[%d] Complete\n", this->task_id);
      // print_matrix(c_ptr, a_nrows, b_ncols, "C aft");

      if (beta != 0.) {
        free(tmp_ptr);

        // printf("PASS: free tmp_ptr\n");
      }
    }

    FBLAS_UINT size() {
      FBLAS_UINT a_mem = a_nrows * a_ncols * sizeof(FPTYPE);
      FBLAS_UINT b_mem = a_ncols * b_ncols * sizeof(FPTYPE);
      FBLAS_UINT c_mem = a_nrows * b_ncols * sizeof(FPTYPE);

      return (a_mem + b_mem + c_mem);
    }
    friend class Cache;
  };
}  // namespace flash
