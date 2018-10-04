% ctype = dtype_to_c_type(data_type)

__kernel void split(const int N, __global const <%= ctype %> *A, __global <%= ctype %> *C) {
    // Get the index of the current element to be processed
    const int globalCol = get_global_id(0); // Col ID of C (0..N)
    const int localCol = get_local_id(1);
    int index = N * globalCol;
}