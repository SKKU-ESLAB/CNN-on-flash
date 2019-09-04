echo "create"
../bin/dense_create $1x$2_a.mtx $1 $2 s
../bin/dense_create $2x$3_b.mtx $1 $2 s
../bin/dense_create $1x$3_c.mtx $1 $2 s

echo "gemm-time"
../bin/gemm_driver $1x$2_a.mtx $2x$3_b.mtx $1x$3_c.mtx $1 $2 $3 1 1 N N R $2 $3 $3

