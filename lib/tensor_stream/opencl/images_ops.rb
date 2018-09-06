require 'oily_png'

module TensorStream
  module OpenCLHelpers
    module ImagesOps
      def ImagesOps.included(klass)
        klass.class_eval do
          register_op :decode_png do |context, tensor, inputs|
            content = _run(inputs[0], context)
            channels = tensor.options[:channels]
            channels = 4 if channels.zero?

            image = ChunkyPNG::Image.from_blob(content.buffer.to_a.pack('C*'))
            output_buffer = _create_result_buffer(tensor.data_type, [image.height, image.width, channels], "out_#{tensor.name}")

            image.grayscale! if channels == 1
            image.pixels.each_with_index do |pixel, index|
              start_index = index * channels
              if channels == 4
                output_buffer.buffer[start_index] = ChunkyPNG::Color.r(pixel)
                output_buffer.buffer[start_index + 1] = ChunkyPNG::Color.g(pixel)
                output_buffer.buffer[start_index + 2] = ChunkyPNG::Color.g(pixel)
                output_buffer.buffer[start_index + 3] = ChunkyPNG::Color.a(pixel)
              elsif channels == 3
                output_buffer.buffer[start_index] = ChunkyPNG::Color.r(pixel)
                output_buffer.buffer[start_index + 1] = ChunkyPNG::Color.g(pixel)
                output_buffer.buffer[start_index + 2] = ChunkyPNG::Color.g(pixel)
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
            height, width, channels = shape_eval(image_data)

            png = ChunkyPNG::Image.new(width, height)
            image_data.each_with_index do |rows, h_index|
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
            png.to_s
          end
        end
      end
    end
  end
end