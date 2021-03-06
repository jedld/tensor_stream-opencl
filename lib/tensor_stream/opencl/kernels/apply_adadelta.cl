% c_dtype = dtype_to_c_type(dtype)
 // same dimension add floating point op
 __kernel void apply_adadelta_<%= dtype %>(
                                       __global const <%= c_dtype %> *lr,
                                       __global const <%= c_dtype %> *rho,
                                       __global const <%= c_dtype %> *epsilon,
                                       __global const <%= c_dtype %> *grad,
                                       __global <%= c_dtype %> *output,
                                       __global <%= c_dtype %> *acc,
                                       __global <%= c_dtype %> *acc_update
                                       ) {
    // Get the index of the current element to be processed
    const int index = get_global_id(0);

    acc[index] = acc[index] * rho[0] + (grad[index] * grad[index]) * ((<%= c_dtype %>)1 - rho[0]);
    const <%= c_dtype %> update = sqrt(acc_update[index] + epsilon[0]) * rsqrt(acc[index] + epsilon[0]) * grad[index];
    output[index] -= update * lr[0];
    acc_update[index] = acc_update[index] * rho[0] + update * update * ((<%= c_dtype %>)1 - rho[0]);
}