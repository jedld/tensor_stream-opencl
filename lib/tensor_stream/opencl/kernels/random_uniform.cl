% c_dtype = dtype_to_c_type(dtype)
__kernel void random_uniform_<%= dtype %>(const int seed_ptr, const float min, const float max, __global const <%= c_dtype %> *rand_table, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int id = get_global_id(0);
    <%= c_dtype %> rand_value = rand_table[ (seed_ptr + id) % <%= tsize %>];
    C[id] = rand_value * (max - min) + min;
}