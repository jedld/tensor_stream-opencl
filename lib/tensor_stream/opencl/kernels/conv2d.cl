% ctype = dtype_to_c_type(dtype)

__kernel void conv2d(const int height, const int width, const int out_height, const int out_width, __global const <%= ctype %> *images, __global const <%= ctype %> *filter, __global <%= ctype %> *output) {
    // Get the index of the current element to be processed
    const int batch_index = get_global_id(0);
    const int h_index = get_global_id(1);
    const int w_index = get_global_id(2);
    const int h_index_with_stride = h_index * <%= stride[0] %> - <%= padding[0] %>;
    const int w_index_with_stride = w_index * <%= stride[1] %> - <%= padding[1] %>;

    const int image_index = batch_index * height * width * <%= ch %>;
    const int image_row_width = width * <%= ch %>;
    const int out_image_row_size = out_height * out_width * <%= out_ch %>;

    for (int out_channel_index = 0; out_channel_index < <%= out_ch %>; out_channel_index++) {
      <%= ctype %> sum = 0;
      for (int channel_index = 0; channel_index < <%= ch %>; channel_index++) {
        for(int y = 0; y < <%= fh %>; y++) {
          for (int x = 0; x < <%= fw %>; x++) {
            if ( (h_index_with_stride + y) < height && (w_index_with_stride + x) < width &&
                 (h_index_with_stride + y) >= 0 && (w_index_with_stride + x) >=0) {
              <%= ctype %> f = filter[y*<%= fw * ch * out_ch %> + x*<%= ch * out_ch %> + (channel_index*<%= out_ch %>) + out_channel_index];
              sum += images[image_index + (h_index_with_stride + y)*image_row_width + (w_index_with_stride + x)*<%= ch %> + channel_index] * f;
            }
          }
        }
      }
      output[batch_index * out_image_row_size  + h_index * out_width * <%= out_ch %> +  w_index * <%= out_ch %> + out_channel_index ] = sum;
    }
}