% c_dtype = dtype_to_c_type(dtype)

__kernel void bias_add_grad_<%= dtype %>(__global const <%= c_dtype %> *received_grad, __global <%= c_dtype %> *output) {
    const int id = get_global_id(0);
    <%= c_dtype %> sum = 0;
    for(int i = 0; i < <%= rows %>; i++) {
      sum += received_grad[<%= n %> * i + id];
    }
    output[id] = sum;
}