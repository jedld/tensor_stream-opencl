% c_dtype = dtype_to_c_type(dtype)
% o_multipliers = o_shape.dup.drop(1).reverse.inject([1]) { |a, s| a << s * a.last }.reverse
% i_multipliers = shape.dup.drop(1).reverse.inject([1]) { |a, s| a << s * a.last }.reverse
% out_ops = o_multipliers.map.with_index { |m, index| "id_#{index} * #{m}" }.join(' + ')
% in_axis_multipliers = i_multipliers.select.with_index { |m, index| axis.include?(index) }
% in_axis_ops =  in_axis_multipliers.map.with_index { |m, index| "i_#{index} * #{m}"}.join(' + ')
% in_output_multipliers = i_multipliers.reject.with_index { |m, index| axis.include?(index) }
% in_output_ops =  in_output_multipliers.map.with_index { |m, index| "id_#{index} * #{m}"}.join(' + ')
__kernel void reduce_axis_<%= dtype %>(__global const <%= c_dtype %> *value, __global <%= c_dtype %> *output) {
    // Get the index of the current element to be processed
<% o_multipliers.size.times.each_with_index do |s, index| %>
  const int id_<%= index %> = get_global_id(<%= index %>);
<% end %>

<%= c_dtype %> sum = <%= f == :prod ? 1 : 0 %>;
<%= c_dtype %> item_size = 0;
<% axis.each_with_index do |axis, index| %>
  for (int i_<%= index %> = 0; i_<%= index %> < <%= shape[axis] %>; i_<%= index %>++) {
<% end %>
  int index = <%= in_axis_ops %>;
  item_size += 1;
  <% unless in_output_ops.empty? %>
  index += <%= in_output_ops %>;
  <% end %>
  <%= case(f)
    when :sum, :mean
      "sum += value[index];"
    when :prod
      "sum *= value[index];"
    else
    raise "unkown redunction func #{f}"
    end
  %>
<% axis.each do |axis| %>
  }
<% end %>
<% if f == :mean %>
  output[<%= out_ops %>] = sum / item_size;
<% else %>
  output[<%= out_ops %>] = sum;
<% end %>
}