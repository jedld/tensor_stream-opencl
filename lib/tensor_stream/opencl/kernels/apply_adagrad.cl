% c_dtype = dtype_to_c_type(dtype)
 // same dimension add floating point op
 __kernel void apply_adagrad_<%= dtype %>(
                                        __global const <%= c_dtype %> *lr,
                                       __global const <%= c_dtype %> *grad,
                                       __global <%= c_dtype %> *output,
                                       __global <%= c_dtype %> *acc
                                       ) {
    // Get the index of the current element to be processed
    const int index = get_global_id(0);
    output[index] -= grad[index] * lr[0] * rsqrt(acc[index]);
 }