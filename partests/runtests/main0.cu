extern "C" __global__ void main0 ( int* z , int* z_length , int* x , int* x_length , int* y , int* y_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 if (! ( i < (*z_length) && i < (*x_length) && i < (*y_length) ) ) {
 return ;
 } z [i] = ( x [i] ) + ( y [i] ) ;
 } 