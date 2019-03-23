% c_dtype = dtype_to_c_type(dtype)
% out_c_dtype = dtype_to_c_type(out_dtype)
__kernel void argmin_<%= dtype %>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    <%= c_dtype %> min = <%= max_value_for(dtype) %>;
    <%= out_c_dtype %> min_index = 0;

    for(int i = 0; i < <%= n %>; i++) {
        if (A[i] < min) {
            min = A[i];
            min_index = i;
        }
    }
    C[0] = min_index;
}