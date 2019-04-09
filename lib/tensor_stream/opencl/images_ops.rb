# require 'oily_png'
module TensorStream
  module OpenCLHelpers
    module ImagesOps
      def ImagesOps.included(klass)
        klass.class_eval do
          register_op :decode_png do |context, tensor, inputs|
            content = _run(inputs[0], context)
            channels = tensor.options[:channels]
            resample_new_shape = tensor.options[:new_shape]
            resample_method = tensor.options[:resample_method] || :bilinear
            channels = 4 if channels.zero?

            image = ChunkyPNG::Image.from_blob(content.buffer.to_a.pack('C*'))

            if resample_new_shape
              case resample_method
              when :bilinear
                image.resample_bilinear!(resample_new_shape[1], resample_new_shape[0]) # width, # height
              when :nearest_neighbor
                image.resample_nearest_neighbor!(resample_new_shape[1], resample_new_shape[0])
              else
                raise TensorStream::ValueError, "invalid resample method provided #{resample_method}. Available (:bilinear, :nearest_neighbor)"
              end
            end

            output_buffer = _create_result_buffer(tensor.data_type, [image.height, image.width, channels], "out_#{tensor.name}", allocate_host: true)

            image.grayscale! if channels == 1
            image.pixels.each_with_index do |pixel, index|
              start_index = index * channels
              if channels == 4
                output_buffer.buffer[start_index] = ChunkyPNG::Color.r(pixel)
                output_buffer.buffer[start_index + 1] = ChunkyPNG::Color.g(pixel)
                output_buffer.buffer[start_index + 2] = ChunkyPNG::Color.b(pixel)
                output_buffer.buffer[start_index + 3] = ChunkyPNG::Color.a(pixel)
              elsif channels == 3
                output_buffer.buffer[start_index] = ChunkyPNG::Color.r(pixel)
                output_buffer.buffer[start_index + 1] = ChunkyPNG::Color.g(pixel)
                output_buffer.buffer[start_index + 2] = ChunkyPNG::Color.b(pixel)
              elsif channels == 1
                output_buffer.buffer[start_index] = ChunkyPNG::Color.r(pixel)
              else
                raise "Invalid channel value #{channels}"
              end
            end

            write_op = _opencl_queue.enqueue_write_buffer(output_buffer.cl_buffer, output_buffer.buffer)
            output_buffer.op = write_op
            output_buffer
          end

          register_op :encode_png do |_context, tensor, inputs|
            image_data = inputs[0]

            resample_new_shape = tensor.options[:new_shape]
            resample_method = tensor.options[:resample_method] || :bilinear

            height, width, channels = image_data.shape
            image_buffer = image_data.buffer.reshape(*image_data.shape.reverse).to_a
\
            png = ChunkyPNG::Image.new(width, height)
            image_buffer.each_with_index do |rows, h_index|
              rows.each_with_index do |p_data, w_index|
                if channels == 4
                  png[w_index, h_index] = ChunkyPNG::Color.rgba(p_data[0], p_data[1], p_data[2], p_data[3])
                elsif channels == 3
                  png[w_index, h_index] = ChunkyPNG::Color.rgb(p_data[0], p_data[1], p_data[2])
                elsif channels == 1
                  png[w_index, h_index] = ChunkyPNG::Color.rgb(p_data[0], p_data[0], p_data[0])
                end
              end
            end

            if resample_new_shape
              case resample_method
              when :bilinear
                png.resample_bilinear!(resample_new_shape[1], resample_new_shape[0]) # width, # height
              when :nearest_neighbor
                png.resample_nearest_neighbor!(resample_new_shape[1], resample_new_shape[0])
              else
                raise TensorStream::ValueError, "invalid resample method provided #{resample_method}. Available (:bilinear, :nearest_neighbor)"
              end
            end

            convert_to_opencl(png.to_s, [], data_type: :string, name: tensor.name)
          end
        end
      end
    end
  end
end