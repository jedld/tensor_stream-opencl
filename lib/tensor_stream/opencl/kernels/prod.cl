% c_dtype = dtype_to_c_type(dtype)
__kernel void prod_<%= dtype %>(__global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    int id = get_global_id(0);
    int offset = (id + <%= index %>) * <%= w %>;
    <%= c_dtype %> prod = 1;
    <% if n > 4 %>
      for(int i = 0; i < <%= n/4 %> ; i++) {
        <% sums = 4.times.map do |i|
          "A[offset + #{i}]"
        end %>
        prod *= <%= sums.join(' * ') %>;
        offset += 4;
      }
      <% if n%4!=0 %>
        <% (n % 4).times do |i| %>
          prod *= A[offset + <%= i %>];
        <% end %>
      <% end %>
    <% else %>
      <% n.times do |i| %>
        prod *= A[offset + <%= i %>];
      <% end %>
    <% end %>
    C[id] = prod;
}