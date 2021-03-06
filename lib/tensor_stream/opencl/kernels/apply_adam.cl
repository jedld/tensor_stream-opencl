% c_dtype = dtype_to_c_type(dtype)
 // same dimension add floating point op
 __kernel void apply_adam_<%= dtype %>(
                                       __global const <%= c_dtype %> *grad,
                                       __global const <%= c_dtype %> *learning_rate,
                                       __global const <%= c_dtype %> *beta1_power,
                                       __global const <%= c_dtype %> *beta2_power,
                                       __global const <%= c_dtype %> *beta1,
                                       __global const <%= c_dtype %> *beta2,
                                       __global const <%= c_dtype %> *epsilon,
                                       __global <%= c_dtype %> *momentum,
                                       __global <%= c_dtype %> *output, __global <%= c_dtype %> *v) {
    // Get the index of the current element to be processed
    const int index = get_global_id(0);
    <%= c_dtype %> alpha = learning_rate[0] * sqrt((<%= c_dtype %>)1.0 - beta2_power[0]) / (1.0 - beta1_power[0]);

    momentum[index] += (grad[index] - momentum[index]) * (1.0 - beta1[0]);
    v[index] += (grad[index] * grad[index] - v[index]) * (1.0 - beta2[0]);
    output[index] -= (momentum[index] * alpha) / ( sqrt((<%= c_dtype %>)v[index]) + epsilon[0] );
}