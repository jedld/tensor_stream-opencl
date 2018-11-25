% ctype = dtype_to_c_type(dtype)

__kernel void conv2d_backprop_filter(const int batch_size, const int height, const int width, __global const <%= ctype %> *images, __global const <%= ctype %> *grad, __global <%= ctype %> *output) {
    // Get the index of the current element to be processed
    const int fh_index = get_global_id(0);
    const int fw_index = get_global_id(1);
    const int f_out_channel = get_global_id(2);
    const int image_size = height * width * <%= ch %>;
    const int grad_image_row_width = ( width / <%= stride[1] %>) * <%= out_ch %>;
    const int grad_image_size = (height / <%= stride[0] %>) * (width / <%= stride[1] %>) * <%= out_ch %>;

    for(int channel = 0; channel < <%= ch %>; channel++) {
      <%= ctype %> grad_sum = 0.0;
      for(int batch = 0; batch < batch_size; batch++) {
        int image_index = batch * grad_image_size;
        for(int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            int y_offset = y - fh_index + <%= padding[0] %>;
            int x_offset = x - fw_index + <%= padding[1] %>;
            int y_offset_end = y + (<%= fh %> - fh_index - 1) - <%= padding[2] %>;
            int x_offset_end = x + (<%= fw %> - fw_index - 1) - <%= padding[3] %>;

            if ( (y_offset % <%= stride[0]%>) == 0
              && (x_offset % <%= stride[1]%>) == 0
              && (y_offset >=0) && (x_offset >= 0)
              && (y_offset_end < height)
              && (x_offset_end < width)) {
              <%= ctype %> image_grad = grad[image_index + (y_offset / <%= stride[0] %>) * grad_image_row_width + ( x_offset / <%= stride[1] %>) * <%= out_ch %> + f_out_channel];
              grad_sum += images[batch * image_size + y * width * <%= ch %> + x * <%= ch %> + channel] * image_grad;
            }
          }
        }
      }
      output[fh_index * <%= fw * ch * out_ch %> + fw_index * <%= ch * out_ch %> + channel * <%= out_ch %> + f_out_channel] = grad_sum;
    }
}