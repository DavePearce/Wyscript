extern "C" __global__ void main0 ( int* list , int* list_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 if ( i < ( * list_length ) && i >= 0 ) {
 ( list [i] ) = ( * list_length ) ;
 } } 