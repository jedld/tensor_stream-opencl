% ctype = dtype_to_c_type(dtype)

__kernel void conv2d_backprop_input(const int height, const int width, const int out_height, const int out_width, __global const <%= ctype %> *filter, __global const <%= ctype %> *grad, __global <%= ctype %> *output) {
    // Get the index of the current element to be processed
    int batch_index = get_global_id(0);
    int h_index = get_global_id(1); // orig image y
    int w_index = get_global_id(2); // orig image x

    int h_index_with_stride = h_index / <%= stride[0] %>;
    int w_index_with_stride = w_index / <%= stride[1] %>;
    int grad_height = out_height;
    int grad_width = out_width;

    int image_index = batch_index * grad_height * grad_width * <%= out_ch %>;
    int image_row_width = grad_width * <%= out_ch %>;

    for (int channel_index = 0; channel_index < <%= ch %>; channel_index++) {
      <%= ctype %> g = 0.0;
      for (int out_channel_index = 0; out_channel_index < <%= out_ch %>; out_channel_index++) {
        for(int y = 0; y < <%= fh %>; y++) {
          for (int x = 0; x < <%= fw %>; x++) {
            int y_offset = h_index - y + <%= padding[0] %>;
            int x_offset = w_index - x + <%= padding[1] %>;

            if ( ( y_offset >= 0) && (x_offset >= 0) &&
                 ( y_offset % <%= stride[0]%> == 0) &&
                 ( x_offset % <%= stride[1]%> == 0) &&
                 ( h_index + (<%= fh %> - y - 1) < (height + <%= padding[2] %>)) &&
                 ( w_index + (<%= fw %> - x - 1) < (width + <%= padding[3] %>))
                 ) {
              <%= ctype %> imag_grad = grad[image_index + ( y_offset / <%= stride[0] %>) * image_row_width + ( x_offset / <%= stride[1] %>) * <%= out_ch %> + out_channel_index];
              g += imag_grad * filter[y * <%= fw * ch * out_ch %> + x * <%= ch * out_ch %> + (channel_index*<%= out_ch %>) + out_channel_index];
            }
          }
        }
      }

      output[batch_index * height * width * <%= ch %> + h_index * width * <%= ch %> +  w_index * <%= ch %> + channel_index ] = g;
    }
}