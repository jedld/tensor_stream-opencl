% c_dtype = dtype_to_c_type(dtype)
__kernel void relu6_<%= dtype %>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int id = get_global_id(0);

    C[id] = min((<%= c_dtype %>)max((<%= c_dtype %>) A[id], (<%= c_dtype %>)0), (<%= c_dtype %>)6);
}