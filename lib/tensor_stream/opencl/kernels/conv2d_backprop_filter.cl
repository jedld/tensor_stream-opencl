% ctype = dtype_to_c_type(dtype)

__kernel void conv2d_backprop_filter(const int batch_size, const int height, const int width, __global const <%= ctype %> *images, __global const <%= ctype %> *grad, __global <%= ctype %> *output) {
    // Get the index of the current element to be processed
    const int fh_index = get_global_id(0);
    const int fw_index = get_global_id(1);
    const int f_out_channel = get_global_id(2);
    const int image_size = height * width * <%= ch %>;
    const int grad_image_row_width = width * <%= out_ch %>;
    
    for(int channel = 0; channel < <%= ch %>; channel++) {
      <%= ctype %> grad_sum = 0.0;
      for(int batch = 0; batch < batch_size; batch++) {
        const int image_index = batch * height * width * <%= out_ch %>;
        for(int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            if ( ((y - fh_index) % <%= stride[0]%>) == 0  && ((x - fw_index) % <%= stride[1]%>) == 0 && fh_index <= y && fw_index <= x) {
              const <%= ctype %> image_grad = grad[image_index + ((y + fh_index) / <%= stride[0] %>) * grad_image_row_width + ((x + fw_index) / <%= stride[1] %>) * <%= out_ch %> + f_out_channel];
              grad_sum += images[batch * image_size + y * width * <%= ch %> + x * <%= ch %> + channel] * image_grad;
            }
          }
        }
      }
      output[fh_index * <%= fw * ch * out_ch %> + fw_index * <%= ch * out_ch %> + channel * <%= out_ch %> + f_out_channel] = grad_sum;
    }
}