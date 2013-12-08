extern "C" __global__ void main0 ( int* c , int* c_length , int* a , int* a_length , int* b , int* b_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 if ( i < ( * a_length ) && i >= 0 ) {
 ( c [i] ) = ( ( a [i] ) ) * ( ( b [i] ) ) ;
 } } 