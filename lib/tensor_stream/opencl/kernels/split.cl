% ctype = dtype_to_c_type(data_type)
% mul_str = mul.each_with_index.collect { |mul, index| "#{mul} * index_map_#{index}" }
__kernel void split(const int N, __global const <%= ctype %> *A, __global <%= ctype %> *C) {
    // Get the index of the current element to be processed
    const int globalCol = get_global_id(0); // Col ID of C (0..N)
    const int localCol = get_global_id(1);
    // compute effective coordinates
    int ptr = localCol;
<% dest.each_with_index do |div, index| %>
    <% if index == axis %>
    int index_map_<%= index %> = (int)floor(ptr / (float)<%= div %>) + globalCol * <%= step %>;
    <% else %>
    int index_map_<%= index %> = (int)floor(ptr / (float)<%= div %>);
    <% end %>
    <% if index < dest.size - 1%>ptr = ptr % <%= div %>;<% end %><% end %>
    C[N*globalCol + localCol] =  A[<%= mul_str.join(" + ") %>];
}