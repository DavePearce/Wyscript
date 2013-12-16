extern "C" __global__ void main0 ( int* n , int* z , int* z_length , int* x , int* x_length , int* y , int* y_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 z [i] = ( x [i] ) + ( y [i] ) ;
 } 