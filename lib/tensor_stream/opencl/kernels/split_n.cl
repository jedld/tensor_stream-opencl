% ctype = dtype_to_c_type(data_type)
% mul_str = mul.each_with_index.collect { |mul, index| "#{mul} * index_map_#{index}" }
__kernel void split(__global const <%= ctype %> *A, __global <%= ctype %> *C) {
    // Get the index of the current element to be processed
    const int globalCol = get_global_id(0); // Col ID of C (0..N)

    // compute effective coordinates
    
    const int steps[] = { <%= steps.join(',') %> };
    const int offsets[] = { <%= offsets.join(',') %> };
    const int shapes[] = { <%= shapes.join(',') %> };
    const int sizes[] =  { <%= sizes.join(', ') %> };
    for(int i = 0; i < sizes[globalCol]; i++) {
        int ptr = i;
<% mul.each_with_index do |div, index| %>
    <% if index == axis %>
    int index_map_<%= index %> = (int)floor(ptr / (float)shapes[globalCol]) + steps[globalCol];
    <% else %>
    int index_map_<%= index %> = (int)floor(ptr / (float)<%= div %>);
    <% end %>
    <% if index < mul.size - 1%>ptr = ptr % <%= div %>;<% end %><% end %>
    C[offsets[globalCol] + i] =  A[<%= mul_str.join(" + ") %>];
    }

}