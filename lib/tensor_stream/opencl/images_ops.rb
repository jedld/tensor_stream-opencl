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

            image = ChunkyPNG::Image.from_blob(content.sync!.buffer.to_a.pack('C*'))

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
            image_data = image.pixels.each_with_index.map do |pixel, index|
              start_index = index * channels
              if channels == 4
                [
                 ChunkyPNG::Color.r(pixel),
                 ChunkyPNG::Color.g(pixel),
                 ChunkyPNG::Color.b(pixel),
                 ChunkyPNG::Color.a(pixel)
                ]
              elsif channels == 3
                [
                  ChunkyPNG::Color.r(pixel),
                  ChunkyPNG::Color.g(pixel),
                  ChunkyPNG::Color.b(pixel)
                ]
              elsif channels == 1
                ChunkyPNG::Color.r(pixel)
              else
                raise "Invalid channel value #{channels}"
              end
            end.flatten

            output_buffer.buffer = NArray.to_na(image_data)
            write_op = _opencl_queue.enqueue_write_buffer(output_buffer.cl_buffer, output_buffer.buffer)
            output_buffer.op = write_op
            output_buffer
          end

          register_op :decode_jpg do |context, tensor, inputs|
            require 'jpeg'

            content = _run(inputs[0], context)
            channels = tensor.options[:channels]
            resample_new_shape = tensor.options[:new_shape]
            resample_method = tensor.options[:resample_method] || :bilinear
            channels = 3 if channels.zero?
            jpeg_buffer = content.sync!.buffer.to_a.pack('C*')

            image = Jpeg::Image.open_buffer(jpeg_buffer)

            source_channels = image.color_info == :gray ? 1 : 3
            output_buffer = _create_result_buffer(tensor.data_type, [image.height, image.width, channels], "out_#{tensor.name}", allocate_host: false)


            image_data = image.raw_data.map do |pixel|
              if source_channels == channels
                pixel
              elsif source_channels = 1 && channels == 3
                [pixel, pixel, pixel]
              elsif source_channels = 3 && channels == 1
                raise TensorStream::ValueError, "color to grayscale not supported for jpg"
              end
            end.flatten

            raise TensorStream::ValueError, "float output not supported for jpg decode" if fp_type?(tensor.data_type)

            output_buffer.buffer = NArray.to_na(image_data)

            raise TensorStream::ValueError, "image size mismatch #{output_buffer.shape.reduce(:*)} != #{output_buffer.buffer.size}" if output_buffer.buffer.size != output_buffer.shape.reduce(:*)
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