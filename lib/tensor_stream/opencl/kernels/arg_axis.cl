% c_dtype = dtype_to_c_type(dtype)
% out_c_dtype = dtype_to_c_type(out_dtype)
% o_multipliers = o_shape.dup.drop(1).reverse.inject([1]) { |a, s| a << s * a.last }.reverse
% i_multipliers = shape.dup.drop(1).reverse.inject([1]) { |a, s| a << s * a.last }.reverse
% out_ops = o_multipliers.map.with_index { |m, index| "id_#{index} * #{m}"}.join(' + ')
% axis = axis[0]
% in_axis_multipliers = i_multipliers.select.with_index { |m, index| axis == index }
% in_axis_ops =  in_axis_multipliers.map.with_index { |m| "i * #{m}"}.join(' + ')
% in_output_multipliers = i_multipliers.reject.with_index { |m, index| axis == index }
% in_output_ops =  in_output_multipliers.map.with_index { |m, index| "id_#{index} * #{m}"}.join(' + ')
__kernel void arg_axis_<%= dtype %>(__global const <%= c_dtype %> *value, __global <%= out_c_dtype %> *output) {
    // Get the index of the current element to be processed
<% o_multipliers.size.times.each_with_index do |s, index| %>
  const int id_<%= index %> = get_global_id(<%= index %>);
<% end %>

<%= c_dtype %> min_or_max_value = <%= f == :argmax ? min_value_for(dtype) : max_value_for(dtype) %>;
int min_or_max_index = 0;

for (int i = 0; i < <%= shape[axis] %>; i++) {

  int index = <%= in_axis_ops %>;

  <% unless in_output_ops.empty? %>
  index += <%= in_output_ops %>;
  <% end %>
  <%= case(f)
    when :argmax
      "if (value[index] > min_or_max_value) {"
    when :argmin
      "if (value[index] < min_or_max_value) {"
    else
    raise "unkown redunction func #{f}"
    end
  %>
     min_or_max_index = i;
     min_or_max_value = value[index];
  }
}

  output[<%= out_ops %>] = (<%= out_c_dtype %>)min_or_max_index;
}