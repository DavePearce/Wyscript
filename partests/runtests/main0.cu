extern "C" __global__ void main0 ( int* x , int* x_length , int* y , int* y_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 int xv = x [i] ;
 int yv = y [i] ;
 y [i] = ( ( xv ) * ( xv ) ) + ( ( xv ) * ( xv ) ) ;
 } 