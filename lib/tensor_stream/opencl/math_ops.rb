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

          register_op :bias_add do |context, tensor, inputs|
            value, bias = inputs
            output_buffer = _create_result_buffer(value.data_type, value.shape, tensor.name)
            result_shape = value.shape.dup
            bias_length = result_shape.pop
            work_group = [result_shape.reduce(:*)]
            event_wait_list = build_event_wait_list([value, bias])
            dtype = tensor.data_type
            output_buffer.op = _cl_program('bias_add', n: bias_length, dtype: dtype)
              .send(:"bias_add_#{dtype}", _opencl_queue, work_group, value.cl_buffer,
                    bias.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer
          end

          register_op :bias_add_grad do |context, tensor, inputs|
            received_grad = inputs[0]
            bias_size = received_grad.shape.last
            output_buffer = _create_result_buffer(received_grad.data_type, [bias_size], tensor.name)
            work_group = [bias_size]

            received_grad_shape = received_grad.shape.dup
            received_grad_shape.pop
            item_rows = received_grad_shape.reduce(:*)
            dtype = tensor.data_type
            output_buffer.op = _cl_program('bias_add_grad', n: bias_size, rows: item_rows, dtype: dtype)
              .send(:"bias_add_grad_#{dtype}", _opencl_queue, work_group, received_grad.cl_buffer,
                    output_buffer.cl_buffer, event_wait_list: build_event_wait_list([received_grad]))
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

          %i[argmin argmax].each do |op|
            register_op op do |context, tensor, inputs|
              value, axis = inputs
              rank = value.shape.size
              axis = 0 if axis.nil?

              axis = axis.is_a?(OpenCLBuffer) ? read_final_result(axis) : axis
              raise TensorStream::InvalidArgumentError, "Expected dimension in the range [#{-rank},#{rank}) but got #{axis}" if axis < -rank || axis >= rank

              reduce_multi_axis(context, tensor, value, axis, 'arg', op.to_sym)
             end
          end

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
              reduce_multi_axis(child_context, tensor, value, axis, 'reduce', func)
            end
          end

          def reduce_multi_axis(child_context, tensor, value, axis, prog, func)
            return value if value.shape.empty?

            rank = value.shape.size

            axis = axis.is_a?(OpenCLBuffer) ? read_final_result(axis) : axis
            axis = [axis] unless axis.is_a?(Array)
            return value if axis.empty?
            # remap negative values
            axis.map! { |axis| axis < 0 ? rank - axis.abs : axis }

            new_shape = value.shape.collect.with_index { |v, index| axis.include?(index) ? nil : v }.compact

            buffer_shape = tensor.options[:keepdims] ? _reduced_shape(value.shape.dup, axis) : new_shape
            output_buffer = _create_result_buffer(tensor.options[:output_type] || tensor.data_type, buffer_shape, tensor.name)

            work_group = new_shape.empty? ? [1] : new_shape
            dtype = value.data_type

            output_buffer.op = _cl_program("#{prog}_axis", f: func, axis: axis, shape: value.shape, o_shape: new_shape, dtype: dtype, out_dtype: tensor.options[:output_type])
                .send("#{prog}_axis_#{dtype}", _opencl_queue, work_group, value.cl_buffer,
                      output_buffer.cl_buffer, event_wait_list: build_event_wait_list([value]))

            output_buffer
          end
        end
      end
    end
  end
end