extern "C" __global__ void main0 ( int* dim , int* mat3 , int* mat3_height , int* mat3_width , int* mat1 , int* mat1_height , int* mat1_width , int* mat2 , int* mat2_height , int* mat2_width ) {
 int j = threadIdx.x + ( blockDim.x * ( ( gridDim.x * blockIdx.y ) + blockIdx.x) ) ;
 if (! ( j < (*mat3_height) * (*mat3_width) && j < (*mat1_height) * (*mat1_width) && j < (*mat2_height) * (*mat2_width) ) ) {
 return ;
 } mat3 [ j ] = ( mat1 [ j ] ) * ( mat2 [ j ] ) ;
 } 