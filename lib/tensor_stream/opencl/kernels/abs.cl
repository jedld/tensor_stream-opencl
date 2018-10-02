% c_dtype = dtype_to_c_type(dtype)
% if TensorStream::Ops::FLOATING_POINT_TYPES.include?(dtype)
__kernel void abs_<%= dtype%>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int id = get_global_id(0); // Row ID of C (0..M)

    C[id] = fabs(A[id]);
}
% else
% %w[int int32].each do |dt|
__kernel void abs_<%= dt %>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int id = get_global_id(0); // Row ID of C (0..M)

    C[id] = fabs((float)A[id]);
}
% end
%end