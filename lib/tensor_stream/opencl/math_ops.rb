module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module MathOps
      def MathOps.included(klass)
        klass.class_eval do
          %i[max min add real_div div sub floor_mod mod mul pow sigmoid_grad squared_difference].each do |op|
            register_op op do |_context, tensor, inputs|
              execute_2_operand_func(op.to_s, tensor, inputs[0], inputs[1])
            end
          end

          register_op :add_n do |_context, tensor, inputs|
            if inputs.size == 1
              inputs[0]
            else
              work_group = if inputs[0].shape.size > 2
                             [ inputs[0].shape.reduce(:*) / inputs[0].shape.last, inputs[0].shape.last]
                           else
                             m, n = inputs[0].shape
                             [m || 1, n || 1]
                           end

              cl_m = OpenCL::Int1.new(work_group[0])
              cl_n = OpenCL::Int1.new(work_group[1])
              cl_switch = OpenCL::Int1.new(0)
              dtype = tensor.data_type

              output_buffer = _create_result_buffer(tensor.data_type, inputs[0].shape, "out_#{tensor.name}")
              inputs_queue = inputs.dup
              a = inputs_queue.pop
              until inputs_queue.empty?
                b = inputs_queue.pop
                event_wait_list = build_event_wait_list([a, b])
                method_call = :"add_#{a.data_type}_#{b.data_type}"
                event = _cl_program('add', a: a.data_type, b: b.data_type, dtype: dtype).send(method_call, _opencl_queue, work_group, cl_m, cl_n, cl_switch, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
                a = output_buffer
                a.op = event
              end

              output_buffer.op = a.op
              output_buffer
            end
          end

          register_op :floor_div do |context, tensor, inputs|
            if fp_type?(tensor.data_type)
              execute_2_operand_func('floor_div', tensor, inputs[0], inputs[1])
            else
              execute_2_operand_func('div', tensor, inputs[0], inputs[1])
            end
          end

          register_op :mat_mul do |_context, tensor, inputs|
            a, b = inputs

            m = a.shape[0]
            n = b.shape[1]
            v = b.shape[0]
            k = a.shape[1]

            if tensor.options[:transpose_a]
              m = a.shape[1]
              k = a.shape[0]
            end

            if tensor.options[:transpose_b]
              n = b.shape[0]
              v = b.shape[1]
            end

            result_shape = [m, n]

            raise "#{tensor.inputs[0].name} rank must be greater than 1" if a.shape.size < 2
            raise "#{tensor.inputs[1].name} rank must be greater than 1" if b.shape.size < 2
            raise "#{tensor.inputs[0].name} unsupported rank" if b.shape.size != 2 || a.shape.size!=2
            raise "incompatible shape sizes for matrix multiplication (#{a.shape[1]} != #{b.shape[0]}) #{a.shape} vs #{b.shape}" if k != v

            dtype = tensor.data_type
            a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
            output_buffer = _create_result_buffer(a.data_type, result_shape, tensor.name)

            cl_m = OpenCL::Int1.new(m)
            cl_n = OpenCL::Int1.new(n)
            cl_k = OpenCL::Int1.new(k)

            event_wait_list = build_event_wait_list([a, b])
            output_buffer.op = _cl_program('gemm', ta: !!tensor.options[:transpose_a], tb: !!tensor.options[:transpose_b], dtype: dtype).send(:"gemm_#{dtype}", _opencl_queue, result_shape, cl_m, cl_n, cl_k, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)

            output_buffer
          end

          %i[sign exp tan acos asin sin cos abs sqrt negate square reciprocal tanh tanh_grad sigmoid log1p round floor ceil log].each do |op|
            register_op op, noop: true do |context, tensor, inputs|
              execute_func(op.to_s, tensor, inputs[0], context)
            end
          end

          %i[sum mean].each do |op|
            register_op op do |context, tensor, inputs|
              reduction(context, tensor, inputs[0], inputs[1], op.to_sym)
            end
          end

          register_op :prod do |context, tensor, inputs|
            if inputs[0].shape == [0]
              convert_to_opencl([1.0], [], data_type: inputs[0].data_type, name: tensor.name)
            else
              reduction(context, tensor, inputs[0], inputs[1], :prod)
            end
          end

          # register_op :argmin, buffer: true do |_context, tensor, inputs|
          #   axis = inputs[1].nil? || inputs[1].buffer.nil? || inputs[1].buffer.empty? ? 0 : inputs[1].buffer
          #   rank = inputs[0].shape.size
          #   raise TensorStream::InvalidArgumentError, "Expected dimension in the range [#{-rank},#{rank}) but got #{axis}" if axis < -rank || axis >= rank

          #   arr = inputs[0].buffer.reshape(*inputs[0].shape.reverse).to_a
          #   op = get_op_with_axis(arr, axis, 0, inputs[0].data_type, ->(a, b) { a < b })
          #   convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
          # end

          # register_op :argmax, buffer: true do |_context, tensor, inputs|
          #   axis = inputs[1].nil? || inputs[1].buffer.nil? || inputs[1].buffer.empty? ? 0 : inputs[1].buffer
          #   rank = inputs[0].shape.size
          #   raise TensorStream::InvalidArgumentError, "Expected dimension in the range [#{-rank},#{rank}) but got #{axis}" if axis < -rank || axis >= rank

          #   arr = inputs[0].buffer.reshape(*inputs[0].shape.reverse).to_a
          #   op = get_op_with_axis(arr, axis, 0, inputs[0].data_type, ->(a, b) { a > b })
          #   convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
          # end

          def reduction(child_context, tensor, value, axis, func)
            if axis.nil?
              value = _run(value, child_context)
              size = value.shape.reduce(:*) || 1
              if value.shape.empty? # for scalars, just return as is
                value
              else
                reduction_threads = 32
                items_per_thread_threshold = 4

                output_buffer = _create_result_buffer(value.data_type, [], tensor.name)
                event_wait_list = build_event_wait_list([value])

                if (size > reduction_threads) && ((size / reduction_threads) > items_per_thread_threshold)
                  items_per_thread = size / reduction_threads
                  extra_items = size % reduction_threads
                  intermediate_output_buffer = _create_result_buffer(value.data_type, [reduction_threads], tensor.name)

                  temp_values = if extra_items.zero?
                                  _cl_program(func, dtype: value.data_type, index: 0, n: items_per_thread, w: items_per_thread).
                                    send(:"#{func}_#{value.data_type}", _opencl_queue, [reduction_threads], value.cl_buffer, intermediate_output_buffer.cl_buffer, event_wait_list: event_wait_list)
                                else
                                  [_cl_program(func, dtype: value.data_type, index: 0, n: items_per_thread, w: items_per_thread).
                                    send(:"#{func}_#{value.data_type}", _opencl_queue, [reduction_threads - 1], value.cl_buffer, intermediate_output_buffer.cl_buffer, event_wait_list: event_wait_list),
                                  _cl_program(func, dtype: value.data_type, index: reduction_threads - 1, n: items_per_thread + extra_items,  w: items_per_thread).send(:"#{func}_#{value.data_type}", _opencl_queue, [1], value.cl_buffer, intermediate_output_buffer.cl_buffer, event_wait_list: event_wait_list)]
                                end
                  output_buffer.op = _cl_program(func, dtype: value.data_type, n: reduction_threads, index: 0, w: 0).send(:"#{func}_#{value.data_type}", _opencl_queue, [1], value.cl_buffer, output_buffer.cl_buffer, event_wait_list: temp_values)
                  output_buffer
                else
                  output_buffer.op = _cl_program(func, dtype: value.data_type, n: size, index: 0, w: 0).send(:"#{func}_#{value.data_type}", _opencl_queue, [1], value.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
                  output_buffer
                end
               end
            else
              return value if value.shape.empty?

              axis = axis.is_a?(OpenCLBuffer) ? read_final_result(axis) : axis
              input = complete_eval(value, child_context)

              value = value.buffer.reshape(*value.shape.reverse)
              rank = input.shape.size - 1

              if axis.is_a?(Array)
                axis.map { |x| rank - x.abs }.sort.reverse_each do |x|
                  value = value.send(func, x.to_i)
                end
              else
                value = value.send(func, rank - axis.abs)
              end

              new_shape = if value.is_a?(NArray)
                            value.shape.reverse
                          else
                            value = [value]
                            []
                          end

              new_shape = _reduced_shape(input.shape.dup, axis) if tensor.options[:keepdims]

              convert_to_opencl(value.flatten, new_shape, data_type: tensor.data_type, name: tensor.name)
            end
          end
        end
      end
    end
  end
end