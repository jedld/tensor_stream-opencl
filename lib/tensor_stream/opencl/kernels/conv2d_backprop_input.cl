% ctype = dtype_to_c_type(dtype)

__kernel void conv2d_backprop_input(const int height, const int width, __global const <%= ctype %> *filter, __global const <%= ctype %> *grad, __global <%= ctype %> *output) {
    // Get the index of the current element to be processed
    const int batch_index = get_global_id(0);
    const int h_index = get_global_id(1);
    const int w_index = get_global_id(2);

    const int h_index_with_stride = h_index / <%= stride[0] %>;
    const int w_index_with_stride = w_index / <%= stride[1] %>;
    const int grad_height = height / <%= stride[0] %>;
    const int grad_width = width / <%= stride[1] %>;

    const int image_index = batch_index * grad_height * grad_width * <%= ch %>;
    const int image_row_width = grad_width * <%= out_ch %>;

     for (int channel_index = 0; channel_index < <%= ch %>; channel_index++) {
      <%= ctype %> g = 0.0;
      for (int out_channel_index = 0; out_channel_index < <%= out_ch %>; out_channel_index++) {
        const <%= ctype %> imag_grad = grad[image_index + h_index_with_stride * image_row_width +  w_index_with_stride * <%= out_ch %> + out_channel_index];

        for(int y = 0; y < <%= fh %>; y++) {
          for (int x = 0; x < <%= fw %>; x++) {
            if (y <= h_index && x <= w_index && ( (h_index - y) % <%= stride[0]%> == 0) && ( (w_index - x) % <%= stride[1]%> == 0)) {
              g += imag_grad * filter[y*<%= fw * ch * out_ch %> + x*<%= ch * out_ch %> + (channel_index*<%= out_ch %>) + out_channel_index];
            } 
          }
        }
      }

      output[batch_index * height * width * <%= ch %> + h_index * width * <%= ch %> +  w_index * <%= ch %> + channel_index ] = g;
    }
}