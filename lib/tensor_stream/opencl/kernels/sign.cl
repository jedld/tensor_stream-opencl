% c_dtype = dtype_to_c_type(dtype)

__kernel void sign_<%= dtype %>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int id = get_global_id(0); 
    <%= c_dtype %> value = A[id];
% if floating_point?(dtype)
    if (isnan(value) || value == 0.0f) {
      C[id] = 0.0;
    } else {
      C[id] = value < 0 ? -1.0 : 1.0;
    }
% else
  if (value == 0) {
    C[id] = 0;
  } else {
    C[id] = value < 0 ? -1 : 1;
  }
% end
}