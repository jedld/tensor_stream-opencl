% c_dtype = dtype_to_c_type(dtype)

__kernel void assign_add_<%= dtype %>_0(const int rows, __global <%= c_dtype %> *variable, __global const <%= c_dtype %> *value) {
    const int id = get_global_id(0);
    variable[id] += value[id];
}

__kernel void assign_add_<%= dtype %>_1(const int rows, __global <%= c_dtype %> *variable, __global const <%= c_dtype %> *value) {
    const int id = get_global_id(0);
    const int row_item = get_global_id(1);

    variable[id * rows + row_item] += value[row_item];
}