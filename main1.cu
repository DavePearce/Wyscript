extern "C" __global__ void main1 ( int* mat1 , int* mat1_height , int* mat1_width , int* width , int* mat3 , int* mat3_height , int* mat3_width , int* mat2 , int* mat2_height , int* mat2_width ) {
 int j = blockIdx.x * blockDim.x + threadIdx.x ;
 mat3 [ j ] = ( mat1 [ j ] ) * ( mat2 [ j ] ) ;
 } 