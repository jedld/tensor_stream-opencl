% ctype = dtype_to_c_type(data_type)
% mul_str = mul.each_with_index.collect { |mul, index| "#{mul} * index_map_#{index}" }
__kernel void split(const int offset, __global const <%= ctype %> *A, __global <%= ctype %> *C) {
    // Get the index of the current element to be processed
    const int globalCol = get_global_id(0); // Col ID of C (0..N)

    // compute effective coordinates
    int ptr = globalCol;
<% div.each_with_index do |div, index| %>
    <% if index == axis %>
    int index_map_<%= index %> = (int)floor(ptr / (float)<%= div %>) + <%= step %>;
    <% else %>
    int index_map_<%= index %> = (int)floor(ptr / (float)<%= div %>);
    <% end %>
    <% if index < div.size - 1%>ptr = ptr % <%= div %>;<% end %><% end %>
    C[offset + globalCol] =  A[<%= mul_str.join(" + ") %>];

}