% c_dtype = dtype_to_c_type(dtype)

__kernel void bias_add_<%= dtype %>(__global const <%= c_dtype %> *value, __global const <%= c_dtype %> *bias, __global <%= c_dtype %> *output) {
    const int id = get_global_id(0);

    for(int i = 0; i < <%= n %>; i++) {
      output[<%= step %> * id + i] = value[<%= step %> * i] + bias[i];
    }
}