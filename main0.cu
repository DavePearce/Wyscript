extern "C" __global__ void main0 ( int* a , int* a_height , int* a_width ) {
 int j = threadIdx.x + ( blockDim.x * ( ( gridDim.x * blockIdx.y ) + blockIdx.x) ) ;
 a [ j ] = ( a [ j ] ) + ( 1 ) ;
 } 