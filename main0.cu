extern "C" __global__ void main0 ( int* y , int* y_length , int* x , int* x_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 if ( i < ( * y_length ) && i >= 0 ) {
 int xv = ( x [i] ) ;
 int yv = ( y [i] ) ;
 ( y [i] ) = ( ( ( xv ) ) * ( ( xv ) ) ) + ( ( ( ( yv ) ) * ( ( yv ) ) ) - ( ( ( xv ) ) + ( ( yv ) ) ) ) ;
 } } 