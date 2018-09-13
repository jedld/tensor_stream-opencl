module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module ArrayOps
      def ArrayOps.included(klass)
        klass.class_eval do
          register_op :expand_dims, buffer: true do |_context, tensor, inputs|
            axis = inputs[1].buffer[0]
            shape = inputs[0].shape.dup
            axis = -axis if axis == shape.size
            new_shape = shape.insert(axis, 1).compact
            new_buf = inputs[0].buffer.reshape(*new_shape.reverse)
            convert_to_opencl(new_buf, new_shape, data_type: inputs[0].data_type, name: tensor.name)
          end

          register_op :fill, buffer: true do |_context, tensor, inputs|
            shape = inputs[0]
            value = inputs[1]

            narray_size = shape.buffer.to_a.reduce(:*) || 1
            cl_buffer = get_cached_buffer(tensor.name, shape.buffer.to_a)

            buffer = if cl_buffer
                       cl_buffer.buffer
                     else
                       allocate_narray_for_type(tensor.data_type, narray_size)
                     end

            buffer.fill!(value.buffer[0])
            convert_to_opencl(buffer, shape.buffer.to_a, data_type: tensor.data_type, name: tensor.name)
          end

          register_op :stack do |_context, tensor, inputs|
            axis = tensor.options[:axis] || 0
            shape = inputs[0].shape
            rank = shape.size + 1
            elem_size = shape.empty? ? 1 : shape.reduce(:*)

            new_shape = [inputs.size]
            shape.inject(new_shape) { |ns, s| ns << s }

            divisors = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            axis = rank + axis if axis < 0
            rotated_shape = Array.new(axis + 1) { new_shape.shift }
            new_shape = rotated_shape.rotate! + new_shape

            output_buffer = _create_result_buffer(tensor.data_type, new_shape, tensor.name)
            multipliers = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            cl_n = OpenCL::Int1.new(elem_size)
            work_group = [elem_size]
            event_wait_list = build_event_wait_list(inputs)
            ops = inputs.each_with_index.map do |input, index|
              cl_index = OpenCL::Int1.new(index)
              _cl_program("pack", data_type: tensor.data_type, divisors: divisors, multipliers: multipliers, axis: axis).pack(_opencl_queue, work_group, cl_n, cl_index, input.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            end
            output_buffer.op = ops
            output_buffer
          end

          register_op :unstack do |context, tensor, inputs|
            value = inputs[0]
            axis = tensor.options[:axis] || 0
            new_shape = value.shape.dup
            rank = new_shape.size - 1

            elem_size = new_shape.empty? ? 1 : new_shape.reduce(:*)

            divisors = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            axis = rank + axis if axis < 0
            rotated_shape = Array.new(axis + 1) { new_shape.shift }
            new_shape = rotated_shape.rotate!(-1) + new_shape

            output_buffer = _create_result_buffer(tensor.data_type, new_shape, tensor.name)
            multipliers = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            cl_n = OpenCL::Int1.new(elem_size)
            work_group = [elem_size]
            event_wait_list = build_event_wait_list(inputs)
            ops = inputs.each_with_index.map do |input, index|
              cl_index = OpenCL::Int1.new(index)
              _cl_program("unpack", data_type: tensor.data_type, divisors: divisors, multipliers: multipliers, axis: axis).unpack(_opencl_queue, work_group, cl_n, cl_index, input.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            end
            output_buffer.op = ops
            synced_buffer = complete_eval(output_buffer, context)

            step = multipliers[0]
            sub_shape = new_shape.dup
            sub_shape.shift

            outputs = Array.new(new_shape[0]) do |index|
              start = index * step
              convert_to_opencl(synced_buffer.buffer[start...start+step], sub_shape, data_type: tensor.data_type)
            end

            TensorStream::Evaluator::OutputGroup.new(outputs, outputs.map(&:data_type))
          end
        end
      end
    end
  end
end