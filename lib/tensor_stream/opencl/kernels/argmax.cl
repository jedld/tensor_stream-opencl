% c_dtype = dtype_to_c_type(dtype)
% out_c_dtype = dtype_to_c_type(out_dtype)
__kernel void argmax_<%= dtype %>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    <%= c_dtype %> max = <%= min_value_for(dtype) %>;
    <%= out_c_dtype %> max_index = 0;

    for(int i = 0; i < <%= n %>; i++) {
        if (A[i] > max) {
            max = A[i];
            max_index = i;
        }
    }
    C[0] = max_index;
}