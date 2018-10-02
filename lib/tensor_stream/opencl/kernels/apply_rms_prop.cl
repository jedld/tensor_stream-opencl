% c_dtype = dtype_to_c_type(dtype)
 // same dimension add floating point op
 __kernel void apply_rms_prop_<%= dtype %>(__global const <%= c_dtype %> *lr,
                                           __global const <%= c_dtype %> *rho,
                                           __global const <%= c_dtype %> *momentum,
                                           __global const <%= c_dtype %> *epsilon,
                                           __global const <%= c_dtype %> *grad,
                                           __global <%= c_dtype %> *output,
                                           __global <%= c_dtype %> *ms,
                                           __global <%= c_dtype %> *mom) {
    // Get the index of the current element to be processed
    const int id = get_global_id(0);
    ms[id] += (grad[id] * grad[id] - ms[id]) * (1.0 - rho[0]);
    mom[id] = mom[id] * momentum[0] + (grad[id] * lr[0]) / sqrt(ms[id] + epsilon[0]);
    output[id] -= mom[id];
 }