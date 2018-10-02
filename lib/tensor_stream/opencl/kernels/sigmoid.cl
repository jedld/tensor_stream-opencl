% c_dtype = dtype_to_c_type(dtype)

__kernel void sigmoid_<%= dtype %>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int id = get_global_id(0);

    C[id] = 1.0f/(1.0f + exp(-A[id]));
}