extern "C" __global__ void main0 ( int* matrix3 , int* matrix3_height , int* matrix3_width , int* matrix1 , int* matrix1_height , int* matrix1_width , int* matrix2 , int* matrix2_height , int* matrix2_width ) {
 int j = threadIdx.x + ( blockDim.x * ( ( gridDim.x * blockIdx.y ) + blockIdx.x) ) ;
 if (! ( j < (*matrix3_height) * (*matrix3_width) && j < (*matrix1_height) * (*matrix1_width) && j < (*matrix2_height) * (*matrix2_width) ) ) {
 return ;
 } for ( int k = 0 ;
 k < ( 3 - 0 ) ;
 k ++ ) {
 matrix3 [ j ] = ( matrix3 [ j ] ) + ( ( matrix1 [ (blockIdx.x + blockIdx.y * gridDim.x) * (* matrix1_width ) + k ] ) * ( matrix2 [ (blockIdx.x + blockIdx.y * gridDim.x) * (* matrix2_width ) + k ] ) ) ;
 } } 