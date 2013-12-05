extern "C" __global__ void main0 ( int* a , int* a_length , int* x , int* y ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 a [i] = ( * x ) + ( * y ) ;
 } 