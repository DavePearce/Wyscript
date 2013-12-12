extern "C" __global__ void main0 ( int* a , int* a_height , int* a_width ) {
 int j = blockIdx.x * blockDim.x + threadIdx.x ;
 a [ j ] = ( a [ j ] ) + ( 1 ) ;
 } 