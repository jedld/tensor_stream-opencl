% c_dtype = dtype_to_c_type(dtype)
__kernel void prod_<%= dtype %>(const int N, __global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    <%= c_dtype %> sum = 1;
    for(int i = 0; i < N; i++) {
      sum *= A[i];
    }
    C[0] = sum;
}