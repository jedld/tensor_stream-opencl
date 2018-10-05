% ctype = dtype_to_c_type(data_type)

__kernel void concat(const int N, const int index, const int step, __global const <%= ctype %> *A, __global <%= ctype %> *C) {
    // Get the index of the current element to be processed
    const int globalCol = get_global_id(0); // Col ID of C (0..N)
    int ptr = globalCol;

    // compute effective coordinates
<% divisors.each_with_index do |div, index| %>
    <% if axis == index %>
        int index_map_<%= index %> = (int)floor(ptr / (float)<%= div %>) + step;
    <% else %>
        int index_map_<%= index %> = (int)floor(ptr / (float)<%= div %>);
    <% end %>
    <% if index < divisors.size - 1%>
        ptr = ptr % <%= div %>;
    <% end %>
<% end %>

    C[<%= multipliers.each_with_index.map { |m, idx| "#{m}*index_map_#{idx}" }.join(' + ') %>] = A[globalCol];
}