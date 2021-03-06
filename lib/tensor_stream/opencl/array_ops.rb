module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module ArrayOps
      def ArrayOps.included(klass)
        klass.class_eval do

          #fast cached 0/1 constant fill
          register_op %i[zeros ones zeros_like ones_like] do |context, tensor, inputs|
            shape = if %i[zeros_like ones_like].include?(tensor.operation)
                      inputs[0].shape
                    elsif !inputs[0].nil?
                      complete_eval(inputs[0], context).buffer.to_a
                    else
                      tensor.shape.shape
                    end
            cache_key = "cons_#{tensor.name}_#{tensor.data_type}_#{shape}"
            @context[:_cache][:_cl_buffers][cache_key] ||= begin
              buffer = OpenCLBuffer.allocate_narray_for_type(tensor.data_type, shape.reduce(:*) || 1)
              if %i[zeros zeros_like].include?(tensor.operation)
                buffer.fill!(0)
              else
                buffer.fill!(1)
              end
              convert_to_opencl(buffer, shape, data_type: tensor.data_type, name: tensor.name)
            end
          end

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

            fill_shape = shape.nil? ? tensor.shape.shape : shape.buffer.to_a
            narray_size = fill_shape.reduce(:*) || 1

            cl_buffer = get_cached_buffer(tensor.name, fill_shape)

            buffer = if cl_buffer
                       cl_buffer.buffer
                     else
                       OpenCLBuffer.allocate_narray_for_type(tensor.data_type, narray_size)
                     end

            buffer.fill!(value.buffer[0])
            convert_to_opencl(buffer, fill_shape, data_type: tensor.data_type, name: tensor.name)
          end

          register_op :split do |context, tensor, inputs|
            value, num_split, axis = inputs
            value_shape = value.shape
            axis = read_final_result(complete_eval(axis, context))
            num_split = read_final_result(complete_eval(num_split, context))

            multipliers = value_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            outputs = if !num_split.is_a?(Array) # scalar split
                        split_target = value_shape[axis]
                        raise TensorStream::ValueError, "#{num_split} does not divide #{split_target} evenly" if split_target % num_split != 0

                        piece_size = split_target / num_split

                        new_shape = value_shape.dup
                        new_shape[axis] = piece_size

                        if axis.zero? # axis zero fast copy path
                          Array.new(num_split) do |index|
                            _create_result_sub_buffer(value, index, tensor.data_type, new_shape, "#{tensor.name}/out_#{index}_#{num_split}")
                          end
                        else
                          # create buffers for each piece
                          work_buffer = _create_result_buffer(tensor.data_type, value_shape, "#{tensor.name}/out")
                          piece_size = new_shape.reduce(:*)
                          work_group = [num_split, piece_size]

                          divisors = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
                            a << s * a.last
                          end.reverse

                          cl_piece_size = OpenCL::Int1.new(piece_size)
                          event_wait_list = build_event_wait_list(inputs)
                          step = value_shape[axis] / num_split
                          event = _cl_program('split', step: step, axis: axis, mul: multipliers, dest: divisors, data_type: tensor.data_type).split(_opencl_queue, work_group,
                                     cl_piece_size,
                                     value.cl_buffer,
                                     work_buffer.cl_buffer,
                                     event_wait_list: event_wait_list)
                          work_buffer.op = event

                          Array.new(num_split) do |index|
                            _create_result_sub_buffer(work_buffer, index, tensor.data_type, new_shape, "#{tensor.name}/out_#{index}_#{num_split}")
                          end
                        end
                      else
                        raise TensorStream::ValueError, "#{num_split} does not divide #{value_shape[axis]} evenly" if num_split.reduce(:+) != value_shape[axis]

                        # compute shapes of individual output buffers
                        new_shapes = num_split.each_with_index.collect do |num, index|
                                       new_shape = value_shape.dup
                                       new_shape[axis] = num
                                       new_shape
                                     end
                        out = []

                        if axis.zero? # axis zero fast copy path
                          start = 0

                          new_shapes.each_with_index do |ns, index|
                            element_count = ns.reduce(:*) || 1
                            region_size_in_bytes = element_count * value.buffer.element_size
                            out << _create_variable_result_sub_buffer(value, index, start, region_size_in_bytes, tensor.data_type, ns, "#{tensor.name}/out_#{index}_#{ns.join('.')}")
                            start += region_size_in_bytes
                          end
                        else
                          # create buffers for each piece
                          work_buffer = _create_result_buffer(tensor.data_type, value_shape, "#{tensor.name}/out")
                          start = 0

                          steps = num_split.dup.reverse.drop(1).inject([0]) do |a, s|
                            a << s + a.last
                          end

                          offsets = new_shapes.dup.reverse.drop(1).inject([0]) do |a, shape|
                            size_bytes = shape.reduce(:*) || 1
                            a << a.last + size_bytes
                          end

                          events = new_shapes.each_with_index.collect do |shape, index|
                            offset = offsets[index]
                            step = steps[index]
                            divisors = shape.dup.drop(1).reverse.inject([1]) do |a, s|
                              a << s * a.last
                            end.reverse
                            piece_size = shape.reduce(:*) || 1
                            work_group = [piece_size]
                            cl_offset = OpenCL::Int1.new(offset)

                            _cl_program('split_n', axis: axis,
                                                           div: divisors,
                                                           mul: multipliers,
                                                           step: step,
                                                           data_type: tensor.data_type).
                                                          split(_opencl_queue,
                                                                work_group,
                                                                cl_offset,
                                                                value.cl_buffer,
                                                                work_buffer.cl_buffer,
                                                                event_wait_list: event_wait_list)
                          end
                          work_buffer.op = events
                          new_shapes.each_with_index do |ns, index|
                            element_count = ns.reduce(:*) || 1
                            region_size_in_bytes = element_count * work_buffer.buffer.element_size
                            out << _create_variable_result_sub_buffer(work_buffer, index, start, region_size_in_bytes, tensor.data_type, ns, "#{tensor.name}/out_#{index}_#{new_shape.join('.')}")
                            start += region_size_in_bytes
                          end
                        end

                        out
                      end

            TensorStream::Evaluator::OutputGroup.new(outputs, outputs.map(&:data_type))
          end

          register_op :concat do |context, tensor, inputs|
            axis = inputs.shift
            shape = inputs[0].shape

            normal_shape = inputs[0].shape.dup

            axis = read_final_result(_run(axis, context))
            axis = normal_shape.size - 1 if axis == -1

            divisors = normal_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            new_shape = inputs[0].shape.dup
            new_shape[axis] = 0
            inputs.each do |input|
              new_shape[axis] += input.shape[axis]
            end

            multipliers = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            output_buffer = _create_result_buffer(tensor.data_type, new_shape, tensor.name)
            ops = if axis.zero? # fast path
                    inputs.each_with_index.map do |input, index|
                      next if input.empty_value?

                      start = index * input.buffer.size * input.buffer.element_size
                      region = [input.buffer.size * input.buffer.element_size, 1, 1]
                      event_wait_list = build_event_wait_list(input)
                      _opencl_queue.enqueue_copy_buffer_rect(input.cl_buffer, output_buffer.cl_buffer,
                            region, dst_origin: [start, 0, 0], event_wait_list: event_wait_list)
                    end.compact
                  else
                    elem_size = shape.empty? ? 1 : shape.reduce(:*)
                    cl_n = OpenCL::Int1.new(elem_size)

                    steps = inputs.map(&:shape).reverse.drop(1).inject([0]) do |a, shape|
                      a << shape[axis] + a.last
                    end

                    work_group = [elem_size]
                    event_wait_list = build_event_wait_list(inputs)

                    inputs.each_with_index.map do |input, index|
                      cl_index = OpenCL::Int1.new(index)
                      step = OpenCL::Int1.new(steps[index])
                      _cl_program('concat', data_type: tensor.data_type, divisors: divisors, multipliers: multipliers, axis: axis).
                                    concat(_opencl_queue, work_group, cl_n, cl_index, step, input.cl_buffer,
                                          output_buffer.cl_buffer, event_wait_list: event_wait_list)
                    end
                  end

            output_buffer.op = ops
            output_buffer
          end

          register_op :squeeze do |_context, tensor, inputs|
            arr = inputs[0]
            shape = inputs[0].shape.dup
            axis = !tensor.options[:axis].is_a?(Array) ? [tensor.options[:axis]] : tensor.options[:axis]
            if !axis.empty?
              axis.each do |x|
                raise TensorStream::ValueError, "unable to squeeze dimension that does not have a size of 1" unless shape[x] == 1

                shape[x] = nil
              end
            else
              shape = shape.map { |s| s == 1 ? nil : s }
            end

            OpenCLBuffer.new(self, name: tensor.name, data_type: tensor.data_type,
                                   shape: shape.compact, buffer: arr.buffer,
                                   cl_buffer: arr.cl_buffer,
                                   op: arr.op)
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

            ops = if axis.zero? # fast path if axis == 0
                    step = multipliers[0]
                    inputs.each_with_index.map do |input, index|
                      start = index * step * input.buffer.element_size
                      region = [input.buffer.size * input.buffer.element_size, 1, 1]
                      _opencl_queue.enqueue_copy_buffer_rect(input.cl_buffer, output_buffer.cl_buffer, region, dst_origin: [start, 0, 0], event_wait_list: input.op)
                    end
                  else
                    event_wait_list = build_event_wait_list(inputs)
                    inputs.each_with_index.map do |input, index|
                      cl_index = OpenCL::Int1.new(index)
                      _cl_program('pack', data_type: tensor.data_type, divisors: divisors, multipliers: multipliers, axis: axis).pack(_opencl_queue, work_group, cl_n, cl_index, input.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
                    end
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

            multipliers = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
              a << s * a.last
            end.reverse

            sub_shape = new_shape.dup
            sub_shape.shift

            outputs = if axis.zero? # shortcut for axis == 0
                        Array.new(new_shape[0]) do |index|
                          _create_result_sub_buffer(value, index, tensor.data_type, sub_shape, "#{tensor.name}/out_#{index}")
                        end
                      else
                        output_buffer = _create_result_buffer(tensor.data_type, new_shape, tensor.name)
                        cl_n = OpenCL::Int1.new(elem_size)
                        work_group = [elem_size]
                        event_wait_list = build_event_wait_list(inputs)
                        ops = inputs.each_with_index.map do |input, index|
                          cl_index = OpenCL::Int1.new(index)
                          _cl_program('unpack', data_type: tensor.data_type, divisors: divisors, multipliers: multipliers, axis: axis).unpack(_opencl_queue, work_group, cl_n, cl_index, input.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
                        end
                        output_buffer.op = ops
                        Array.new(new_shape[0]) do |index|
                          _create_result_sub_buffer(output_buffer, index, tensor.data_type, sub_shape, "#{tensor.name}/out_#{index}")
                        end
                      end

            TensorStream::Evaluator::OutputGroup.new(outputs, outputs.map(&:data_type))
          end

          register_op :index, noop: true do |context, tensor, inputs|
            a = _run(inputs[0], context)
            index = inputs[1].value || read_final_result(_run(inputs[1], context))

            if a.is_a?(TensorStream::Evaluator::OutputGroup)
              a.outputs[index]
            elsif a.is_a?(Array)
              a[index]
            else
              new_shape = a.shape.dup
              new_shape.shift
              _create_result_sub_buffer(a, index, tensor.data_type, new_shape, "#{tensor.name}/out_#{index}")
            end
          end

          register_op :shape do |_context, tensor, inputs|
            wrap_opencl(inputs[0].shape, name: tensor.name, data_type: tensor.data_type)
          end

          register_op :shape_n do |_context, tensor, inputs|
            shapes = inputs.collect.with_index do |input, index|
              wrap_opencl(input.shape, name: "#{tensor.name}_#{index}", data_type: tensor.data_type)
            end
            TensorStream::Evaluator::OutputGroup.new(shapes, shapes.map { tensor.data_type })
          end

          register_op :reshape do |context, tensor, inputs|
            arr, new_shape = inputs
            new_shape = complete_eval(new_shape, context).buffer.to_a

            shape = if new_shape.size.zero? && arr.buffer.size == 1
                      new_shape
                    else
                      TensorShape.fix_inferred_elements(new_shape, arr.buffer.size)
                    end

            OpenCLBuffer.new(self, name: tensor.name, data_type: tensor.data_type,
                                   shape: shape, buffer: arr.buffer,
                                   cl_buffer: arr.cl_buffer,
                                   op: arr.op)
          end

          register_op :transpose, buffer: true do |_context, tensor, inputs|
            t_param = Array.new(inputs[0].shape.size) { |index| index }.reverse

            if inputs[0].shape.size == 2 && inputs[1].nil?
              transposed = inputs[0].buffer.reshape(*inputs[0].shape.reverse).transpose(*t_param)
              res = convert_to_opencl(transposed.flatten, transposed.shape.reverse, data_type: inputs[0].data_type, name: tensor.name)
              res
            else
              rank = inputs[0].shape.size
              perm = inputs[1].nil? ? (0...rank).to_a.reverse : inputs[1].buffer!
              new_shape = perm.map { |p| inputs[0].shape[p] }.to_a
              output_buffer = _create_result_buffer(tensor.data_type, new_shape, tensor.name, allocate_host: true)
              transpose_with_perm(inputs[0].buffer, output_buffer.buffer, inputs[0].shape, new_shape, perm)

              write_op = _opencl_queue.enqueue_write_buffer(output_buffer.cl_buffer, output_buffer.buffer)
              output_buffer.op = write_op
              output_buffer
            end
          end

          register_op :slice, noop: true do |context, tensor, inputs|
            input_a = complete_eval(inputs[0], context)
            input_b = read_final_result(complete_eval(inputs[1], context))
            size = tensor.options[:size]

            shape = input_a.shape

            slice_param = input_b.zip(size).collect.with_index do |p, index|
              p[1] = p[1] == -1 ? shape[index] : p[1]
              p[0]..p[0] + p[1] - 1
            end.reverse

            new_buf = input_a.buffer.reshape(*input_a.shape.reverse)
            sliced = new_buf.slice[*slice_param]
            convert_to_opencl(sliced.flatten, sliced.shape.reverse, data_type: inputs[0].data_type, name: tensor.name)
          end

          register_op :rank do |_context, tensor, inputs|
            wrap_opencl(inputs[0].shape.size, data_type: tensor.data_type, name: tensor.name)
          end

          register_op :cast do |_context, tensor, inputs|
            a = inputs[0]
            if a.data_type != tensor.data_type
              buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)
              work_group = if inputs[0].shape.size > 2
                             [inputs[0].shape.reduce(:*) / inputs[0].shape.last, inputs[0].shape.last]
                           else
                             m, n = inputs[0].shape
                             [m || 1, n || 1]
                           end

              cl_m = OpenCL::Int1.new(work_group[0])
              cl_n = OpenCL::Int1.new(work_group[1])

              event_wait_list = build_event_wait_list(inputs)
              buffer.op = _cl_program("cast", source_dt: a.data_type, target_dt: tensor.data_type).cast(_opencl_queue, work_group, cl_m, cl_n, a.cl_buffer, buffer.cl_buffer, event_wait_list: event_wait_list)
              buffer
            else
              a
            end
          end

          register_op :range do |context, tensor, inputs|
            start, limit, delta = complete_eval(inputs, context).map { |p| p.buffer.to_a.first }

            if limit.zero?
              limit = start
              start = 0
            end

            raise " delta !=0 " if delta.zero?
            raise " Requires start <= limit when delta > 0" if (start > limit) && delta > 0
            raise " Requires start >= limit when delta < 0" if (start < limit) && delta < 0
            cache_key = "range_#{start}_#{limit}_#{delta}_#{tensor.data_type}"

            @context[:_cache][:_cl_buffers][cache_key] ||= begin
              delta =  fp_type?(tensor.options[:output_type]) ? delta.to_f : delta.to_i
              cur_step = fp_type?(tensor.options[:output_type]) ? start.to_f : start.to_i
              r = []
              Kernel.loop do
                break if start == limit
                break if (start < limit) && (cur_step >= limit)
                break if (start > limit) && (cur_step <= limit)

                r << cur_step
                cur_step += delta
              end
              r
              convert_to_opencl(r, [r.size], data_type: tensor.options[:output_type], name: tensor.name)
            end
          end

          register_op :gather do |context, tensor, inputs|
            params, indexes = inputs
            raise TensorStream::ValueError, "axis !=0 not supported" if tensor.options[:axis] != 0

            new_shape = [indexes.buffer.size]
            target_shape = params.shape.dup
            target_shape.shift
            new_shape += target_shape

            target_shape_size = target_shape.empty? ? 1 : target_shape.reduce(:*)

            output_buffer = _create_result_buffer(tensor.data_type, new_shape, tensor.name)
            event_wait_list = build_event_wait_list(inputs)

            output_buffer.op = indexes.buffer.to_a.map.with_index do |target_index, index|
              source_ptr = target_index * target_shape_size * params.buffer.element_size
              dest_ptr = index * target_shape_size * params.buffer.element_size
              region = [target_shape_size * params.buffer.element_size, 1, 1]
              raise TensorStream::ValueError, "Index out of bounds #{target_index} >= #{params.shape.reduce(:*)}" if target_index >= params.shape.first
              _opencl_queue.enqueue_copy_buffer_rect(params.cl_buffer, output_buffer.cl_buffer, region, src_origin: [source_ptr, 0, 0], dst_origin: [dest_ptr, 0, 0], event_wait_list: event_wait_list) rescue binding.pry
            end
            output_buffer
          end
        end
      end
    end
  end
end