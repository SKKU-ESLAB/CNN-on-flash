// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <chrono>
#include <fstream>
//#include "mkl.h"
#include <cblas.h>
#include "types.h"
#include "utils.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
//#include "utils/Utils.h"

using namespace std::chrono;
using namespace arm_compute;
// using namespace utils;

flash::Logger logger("in_mem");

int main(int argc, char** argv) {
  LOG_ASSERT(logger, argc == 10,
             "Usage Mode : <exec> <mat_A_file> <mat_B_file> <mat_C_file> "
             "<A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <output_file> ");
  //             "<A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <a transpose?> <b
  //             "
  //             "transpose?> <matr order> <lda_a> <lda_b> <lda_c>
  //             <output_file>");

  // map matrices to flash pointers
  std::string A_name = std::string(argv[1]);
  std::string B_name = std::string(argv[2]);
  std::string C_name = std::string(argv[3]);
  // problem dimension
  FBLAS_UINT m = (FBLAS_UINT) std::stol(argv[4]);
  FBLAS_UINT k = (FBLAS_UINT) std::stol(argv[5]);
  FBLAS_UINT n = (FBLAS_UINT) std::stol(argv[6]);

  FPTYPE      alpha = (FPTYPE) std::stof(argv[7]);
  FPTYPE      beta = (FPTYPE) std::stof(argv[8]);
  std::string output_file = std::string(argv[9]);
  //  CHAR       ta = argv[9][0];
  //  CHAR       tb = argv[10][0];
  //  CHAR       ord = argv[11][0];
  //  FBLAS_UINT lda_a = (FBLAS_UINT) std::stol(argv[12]);
  //  FBLAS_UINT lda_b = (FBLAS_UINT) std::stol(argv[13]);
  //  FBLAS_UINT lda_c = (FBLAS_UINT) std::stol(argv[14]);

  float* mat_A = (float*) malloc(m * k * sizeof(float));
  float* mat_B = (float*) malloc(n * k * sizeof(float));
  float* mat_C = (float*) malloc(m * n * sizeof(float));
  float* mat_output = (float*) malloc(m * n * sizeof(float));

  printf("Before init\n");
  printf("mat A: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", mat_A[i]);
  printf("\n");
  printf("mat B: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", mat_B[i]);
  printf("\n");
  printf("mat C: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", mat_C[i]);
  printf("\n");
  printf("mat output: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", mat_output[i]);
  printf("\n");

  LOG_INFO(logger, "Reading matrix A into memory");
  std::ifstream a_file(A_name, std::ios::binary);
  a_file.read((char*) mat_A, m * k * sizeof(float));
  a_file.close();
  LOG_INFO(logger, "Reading matrix B into memory");
  std::ifstream b_file(B_name, std::ios::binary);
  b_file.read((char*) mat_B, k * n * sizeof(float));
  b_file.close();
  LOG_INFO(logger, "Reading matrix C into memory");
  std::ifstream c_file(C_name, std::ios::binary);
  c_file.read((char*) mat_C, m * n * sizeof(float));
  c_file.close();

  Tensor tensor_A, tensor_B, tensor_C;
  Tensor tensor_output;

  LOG_INFO(logger, "Tensor allcator init");
  tensor_A.allocator()->init(TensorInfo(TensorShape(k, m), 1, DataType::F32));
  tensor_B.allocator()->init(TensorInfo(TensorShape(n, k), 1, DataType::F32));
  tensor_C.allocator()->init(TensorInfo(TensorShape(n, m), 1, DataType::F32));
  tensor_output.allocator()->init(
      TensorInfo(TensorShape(n, m), 1, DataType::F32));
  printf("%f %f\n", alpha, beta);

  LOG_INFO(logger, "Tensor allocator import memory");
  tensor_A.allocator()->import_memory(mat_A, m * k * sizeof(float));
  tensor_B.allocator()->import_memory(mat_B, n * k * sizeof(float));
  tensor_C.allocator()->import_memory(mat_C, m * n * sizeof(float));
  tensor_output.allocator()->import_memory(mat_output, m * n * sizeof(float));
  // tensor_output.allocator()->allocate();

  LOG_INFO(logger, "GEMM configure");
  NEGEMM gemm;
  gemm.configure(&tensor_A, &tensor_B, &tensor_C, &tensor_output, alpha, beta);

  LOG_DEBUG(logger, "dimensions : A = ", m, "x", k, ", B = ", k, "x", n);
  LOG_INFO(logger, "Starting sgemm call");

  //  decltype(CblasNoTrans)  trans_a = ta == 'T' ? CblasTrans : CblasNoTrans;
  //  decltype(CblasNoTrans)  trans_b = tb == 'T' ? CblasTrans : CblasNoTrans;
  // decltype(CblasRowMajor) mat_ord = ord == 'R' ? CblasRowMajor :
  // CblasColMajor;
  // execute gemm call

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  gemm.run();
  /*
  mkl_gemm(mat_ord, trans_a, trans_b,          // ordering
           m, n, k,                            // sizes
           alpha, mat_A, lda_a, mat_B, lda_b,  // input
           beta, mat_C, lda_c);                // output
  */
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> span = duration_cast<duration<double>>(t2 - t1);
  LOG_INFO(logger, "gemm() took ", span.count());

  LOG_INFO(logger, "Writing output to file");
  std::ofstream cout_file(output_file, std::ios::binary);
  cout_file.write((char*) tensor_output.buffer(), m * n * sizeof(float));

  cout_file.close();

  // Print result
  printf("Result\n");
  printf("mat A: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", ((float*) tensor_A.buffer())[i]);
  printf("\n");
  printf("mat B: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", ((float*) tensor_B.buffer())[i]);
  printf("\n");
  printf("mat C: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", ((float*) tensor_C.buffer())[i]);
  printf("\n");
  printf("mat output: ");
  for (int i = 0; i < 10; i++)
    printf("%f ", ((float*) tensor_output.buffer())[i]);
  printf("\n");
  /*
  for (int i = 0; i < 10; i++) {
    if (i % m == 0)
      printf("\n");
    printf("%f ", ((float*) tensor_C.buffer())[i]);
  }
  printf("\n\n");
  */

  /*  // Print result
    for (int i = 0; i < m * n; i++) {
      if (i % m == 0)
        printf("\n");
      printf("%f ", mat_output[i]);
    }
    printf("\n");
  */

  // free memory
  free(mat_A);
  free(mat_B);
  free(mat_C);
  free(mat_output);

  return 0;
}
