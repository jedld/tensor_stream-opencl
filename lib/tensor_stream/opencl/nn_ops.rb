module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module NNOps
      def NNOps.included(klass)
        klass.class_eval do

          # Fast in place multiply subtract assign
          register_op :apply_gradient_descent do |_context, tensor, inputs|
            _target_var, learning_rate, delta = inputs

            assign = tensor.inputs[0] || tensor

            assign.buffer.dirty = true # force buffer copy when variable is read externally
            output_buffer = assign.buffer

            work_group = [output_buffer.total_elements]

            event_wait_list = build_event_wait_list([assign.buffer, learning_rate, delta])

            event = call_program("apply_gradient", output_buffer.data_type,
                           work_group,
                           delta.cl_buffer,
                           learning_rate.cl_buffer,
                           output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          # updates for gradient descent with momentum
          register_op :apply_momentum do |_context, tensor, inputs|
            target_var, momentum_var, learning_rate, grad, momentum = inputs

            assign = tensor.inputs[0] || tensor
            assign_acc = tensor.inputs[1]
            assign.buffer.dirty = true # force buffer copy when variable is read externally
            assign_acc.buffer.dirty = true # force buffer copy when variable is read externally

            output_buffer = assign.buffer

            work_group = [output_buffer.total_elements]

            event_wait_list = build_event_wait_list([assign.buffer, assign_acc.buffer, learning_rate, grad, momentum])
            method_call = :"apply_momentum_#{output_buffer.data_type}"
            event = _cl_program("apply_momentum", nesterov: tensor.options[:use_nesterov], dtype: output_buffer.data_type).
                        send(method_call, _opencl_queue, work_group, grad.cl_buffer,
                            learning_rate.cl_buffer, momentum.cl_buffer, output_buffer.cl_buffer,
                            assign_acc.buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            assign_acc.buffer.op = event
            output_buffer
          end

          register_op :apply_adadelta do |context, tensor, inputs|
            _target_var, _accum, _accum_update, lr, rho, epsilon, grad = inputs
            assign = tensor.inputs[0] || tensor
            assign_acc = tensor.inputs[1]
            assign_acc_update = tensor.inputs[2]

            # mark variable buffers as dirty
            assign.buffer.dirty = true # force buffer copy when variable is read externally
            assign_acc.buffer.dirty = true # force buffer copy when variable is read externally
            assign_acc_update.buffer.dirty = true # force buffer copy when variable is read externally

            output_buffer = assign.buffer

            work_group = [output_buffer.total_elements]

            event_wait_list = build_event_wait_list(inputs)
            event = call_program('apply_adadelta', output_buffer.data_type,
                                      work_group,
                                      lr.cl_buffer,
                                      rho.cl_buffer,
                                      epsilon.cl_buffer,
                                      grad.cl_buffer,
                                      assign.buffer.cl_buffer,
                                      assign_acc.buffer.cl_buffer,
                                      assign_acc_update.buffer.cl_buffer,
                                      event_wait_list: event_wait_list)
            output_buffer.op = event
            assign_acc.buffer.op = event
            assign_acc_update.buffer.op = event
            output_buffer
          end

          # Adam optimization algorithm
          register_op :apply_adam do |_context, tensor, inputs|
            _target_var, _m, _v, beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t, grad = inputs

            assign = tensor.inputs[0] || tensor
            assign_m = tensor.inputs[1]
            assign_v = tensor.inputs[2]

            # mark variable buffers as dirty
            assign.buffer.dirty = true # force buffer copy when variable is read externally
            assign_m.buffer.dirty = true # force buffer copy when variable is read externally
            assign_v.buffer.dirty = true # force buffer copy when variable is read externally

            output_buffer = assign.buffer

            work_group = [output_buffer.total_elements]

            event_wait_list = build_event_wait_list(inputs)
            event = call_program("apply_adam", output_buffer.data_type,
                                      work_group,
                                      grad.cl_buffer,
                                      lr_t.cl_buffer,
                                      beta1_power.cl_buffer,
                                      beta2_power.cl_buffer,
                                      beta1_t.cl_buffer,
                                      beta2_t.cl_buffer,
                                      epsilon_t.cl_buffer,
                                      assign_m.buffer.cl_buffer,
                                      assign.buffer.cl_buffer,
                                      assign_v.buffer.cl_buffer,
                                      event_wait_list: event_wait_list)
            output_buffer.op = event
            assign_m.buffer.op = event
            assign_v.buffer.op = event
            output_buffer
          end

          register_op :apply_adagrad do |context, tensor, inputs|
            _target_var, _accum, lr, grad = inputs

            assign = tensor.inputs[0] || tensor
            assign_acc = tensor.inputs[1]

            assign.buffer.dirty = true
            assign_acc.buffer.dirty = true
            output_buffer = assign.buffer

            work_group = [output_buffer.total_elements]

            event_wait_list = build_event_wait_list(inputs)
            event = call_program('apply_adagrad',
                                      output_buffer.data_type,
                                      work_group,
                                      lr.cl_buffer,
                                      grad.cl_buffer,
                                      assign.buffer.cl_buffer,
                                      assign_acc.buffer.cl_buffer,
                                      event_wait_list: event_wait_list)
            output_buffer.op = event
            assign_acc.buffer.op = event
            output_buffer
          end

          register_op :apply_centered_rms_prop do |context, tensor, inputs|
            var, mg, ms, mom, lr, rho, momentum, epsilon, grad = inputs

            assign = tensor.inputs[0]
            assign_mg = tensor.inputs[1]
            assign_ms = tensor.inputs[2]
            assign_mom = tensor.inputs[3]

            assign.buffer.dirty = true
            assign_mg.buffer.dirty = true
            assign_ms.buffer.dirty = true
            assign_mom.buffer.dirty = true
            output_buffer = assign.buffer
            event_wait_list = build_event_wait_list(inputs)
            work_group = [output_buffer.total_elements]

            event = call_program('apply_centered_rms_prop', output_buffer.data_type, work_group,
                            lr.cl_buffer,
                            rho.cl_buffer,
                            momentum.cl_buffer,
                            epsilon.cl_buffer,
                            grad.cl_buffer,
                            assign.buffer.cl_buffer,
                            assign_ms.buffer.cl_buffer,
                            assign_mg.buffer.cl_buffer,
                            assign_mom.buffer.cl_buffer,
                            event_wait_list: event_wait_list)

            output_buffer.op = event
            assign_mg.buffer.op = event
            assign_ms.buffer.op = event
            assign_mom.buffer.op = event
            output_buffer
          end

          register_op :apply_rms_prop do |context, tensor, inputs|
            var, ms, mom, lr, rho, momentum, epsilon, grad = inputs

            assign = tensor.inputs[0]
            assign_ms = tensor.inputs[1]
            assign_mom = tensor.inputs[2]

            assign.buffer.dirty = true
            assign_ms.buffer.dirty = true
            assign_mom.buffer.dirty = true
            output_buffer = assign.buffer
            event_wait_list = build_event_wait_list(inputs)
            work_group = [output_buffer.total_elements]

            event = call_program('apply_rms_prop', output_buffer.data_type,
                            work_group,
                            lr.cl_buffer,
                            rho.cl_buffer,
                            momentum.cl_buffer,
                            epsilon.cl_buffer,
                            grad.cl_buffer,
                            assign.buffer.cl_buffer,
                            assign_ms.buffer.cl_buffer,
                            assign_mom.buffer.cl_buffer,
                            event_wait_list: event_wait_list)

            output_buffer.op = event
            assign_ms.buffer.op = event
            assign_mom.buffer.op = event
            output_buffer
          end

          register_op :softmax do |_context, tensor, inputs|
            a = inputs[0]
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape

            raise "unsupported rank " if a.shape.size > 2

            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("softmax", dtype: dtype).send(:"softmax_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          register_op :log_softmax do |_context, tensor, inputs|
            a = inputs[0] # logits
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape

            raise "unsupported rank " if a.shape.size > 2

            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("log_softmax", dtype: dtype).send(:"log_softmax_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          register_op :softmax_cross_entropy_with_logits_v2 do |context, tensor, inputs|
            a = inputs[0] # logits
            b = inputs[1] # labels
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)
            output_buffer_backprop = _create_result_buffer(tensor.data_type, a.shape, "#{tensor.name}_2")
            rank = a.shape.size - 1
            m, n = a.shape

            raise "unsupported rank " if a.shape.size > 2

            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("softmax_cross", dtype: dtype).send(:"softmax_cross_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, b.cl_buffer,
                                 output_buffer.cl_buffer, output_buffer_backprop.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer_backprop.op = event

            loss = reduction(context, tensor, output_buffer, rank, :sum)
            TensorStream::Evaluator::OutputGroup.new([loss, output_buffer_backprop],  [tensor.inputs[0].data_type, tensor.inputs[0].data_type])
          end

          register_op :softmax_cross_entropy_with_logits_v2_grad do |_context, tensor, inputs|
            a = inputs[0] # logits
            b = inputs[1] # labels
            c = inputs[2] # grads
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape

            raise "unsupported rank " if a.shape.size > 2

            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("softmax_cross_grad", dtype: dtype).send(:"softmax_cross_grad_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, b.cl_buffer, c.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          register_op :sparse_softmax_cross_entropy_with_logits do |context, tensor, inputs|
            a = inputs[0] # logits
            labels = read_final_result(complete_eval(inputs[1], context)) # labels
            labels = last_axis(labels)
            num_classes = a.shape.last

            labels = labels.map do |l|
              one_hot = Array.new(num_classes) { 0 }
              one_hot[l] = 1
              one_hot
            end

            b = wrap_opencl(labels, data_type: inputs[0].data_type, name: "#{tensor.name}_label")

            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)
            output_buffer_backprop = _create_result_buffer(tensor.data_type, a.shape, "#{tensor.name}_2")
            rank = a.shape.size - 1
            m, n = a.shape

            raise "unsupported rank " if a.shape.size > 2

            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("softmax_cross", dtype: dtype).send(:"softmax_cross_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, b.cl_buffer,
                                 output_buffer.cl_buffer, output_buffer_backprop.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer_backprop.op = event

            loss = reduction(context, tensor, output_buffer, rank, :sum)
            TensorStream::Evaluator::OutputGroup.new([loss, output_buffer_backprop],  [tensor.inputs[0].data_type, tensor.inputs[0].data_type])
          end

          register_op :softmax_grad do |_context, tensor, inputs|
            a, grad = inputs

            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape
            raise "unsupported rank " if a.shape.size > 2
            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)
            event = _cl_program('softmax_grad', dtype: dtype, size: n).
                        send(:"softmax_grad_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer,
                             grad.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          %i[relu6].each do |op|
            register_op op, noop: true do |context, tensor, inputs|
              execute_func(op.to_s, tensor, inputs[0], context)
            end
          end

          # Fast per pixel parallel convolution operation
          register_op :conv2d do |_context, tensor, inputs|
            filter = inputs[1]
            batch, height, width, channel = inputs[0].shape
            filter_shape = filter.shape
            strides = tensor.options[:strides]
            height_stride = strides[1]
            width_stride = strides[2]

            raise TensorStream::ValueError, " Current implementation does not yet support strides in the batch and depth dimensions." if strides[0] != 1 || strides[3] != 1

            padding_option = tensor.options[:padding]
            padding = conv2d_padding_options(padding_option, filter_shape, height, width, height_stride, width_stride)
            event_wait_list = build_event_wait_list(inputs)

            f_height, f_width, _in_channels, out_channels = filter_shape
            out_shape = [batch, height / height_stride, width / width_stride, out_channels]
            output_buffer = _create_result_buffer(tensor.data_type, out_shape, tensor.name)

            cl_image_height = OpenCL::Int1.new(height)
            cl_image_width = OpenCL::Int1.new(width)

            work_dimen = [batch, height / height_stride, width / width_stride]

            output_buffer.op = _cl_program("conv2d", dtype: tensor.data_type, fh: f_height, fw: f_width, ch: channel, out_ch: out_channels, stride: [height_stride, width_stride], padding: padding).send(:conv2d, _opencl_queue, work_dimen, cl_image_height, cl_image_width, inputs[0].cl_buffer,
              inputs[1].cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer
          end

          register_op :conv2d_backprop_input do |context, tensor, inputs|
            image_shape, filter, grad = inputs
            filter_shape = filter.shape

            strides = tensor.options[:strides]
            height_stride = strides[1]
            width_stride = strides[2]

            image_shape = read_final_result(complete_eval(image_shape, context))

            event_wait_list = build_event_wait_list(inputs)
            output_buffer = _create_result_buffer(tensor.data_type, image_shape, tensor.name)

            batch, height, width, channels = image_shape
            f_height, f_width, in_channels, out_channels = filter_shape

            padding_option = tensor.options[:padding]
            padding = conv2d_padding_options(padding_option, filter_shape, height, width, height_stride, width_stride)
            work_dimen = [batch, height, width]

            cl_image_height = OpenCL::Int1.new(height)
            cl_image_width = OpenCL::Int1.new(width)

            output_buffer.op = _cl_program("conv2d_backprop_input", dtype: tensor.data_type, fh: f_height, fw: f_width, ch: channels, out_ch: out_channels, stride: [height_stride, width_stride], padding: padding).send(:conv2d_backprop_input, _opencl_queue, work_dimen, cl_image_height, cl_image_width,
              filter.cl_buffer, grad.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer
          end

          register_op :conv2d_backprop_filter do |context, tensor, inputs|
            images, filter_shape, grad = inputs

            event_wait_list = build_event_wait_list(inputs)

            strides = tensor.options[:strides]
            height_stride = strides[1]
            width_stride = strides[2]

            filter_shape = read_final_result(complete_eval(filter_shape, context))
            output_buffer = _create_result_buffer(tensor.data_type, filter_shape, tensor.name)

            batch_size, height, width, channels = images.shape
            f_height, f_width, input_channels, output_channels = filter_shape
            work_dimen = [f_height, f_width, output_channels]

            padding_option = tensor.options[:padding]
            padding = conv2d_padding_options(padding_option, filter_shape, height, width, height_stride, width_stride)

            cl_batch_size = OpenCL::Int1.new(batch_size)
            cl_image_height = OpenCL::Int1.new(height)
            cl_image_width = OpenCL::Int1.new(width)

            output_buffer.op = _cl_program("conv2d_backprop_filter", dtype: tensor.data_type, fh: f_height, fw: f_width, ch: channels, out_ch: output_channels, stride: [height_stride, width_stride], padding: padding ).send(:conv2d_backprop_filter, _opencl_queue, work_dimen, cl_batch_size, cl_image_height, cl_image_width,
              images.cl_buffer, grad.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer
          end

          def conv2d_padding_options(padding_option, filter_shape, height, width, h_stride, w_stride)
            case padding_option
            when 'SAME'
              [
                calc_pad(height, h_stride, filter_shape[0]),
                calc_pad(width, w_stride, filter_shape[1]),
                calc_pad(height, h_stride, filter_shape[0], true),
                calc_pad(width, w_stride, filter_shape[1], true)
              ]
            when 'VALID'
              [0, 0, (filter_shape[0] - 1), (filter_shape[1] - 1)]
            else
              raise TensorStream::ValueError, "Unsupported padding value #{padding_option}, valid values 'SAME', 'VALID'"
            end
          end

          def calc_pad(w, stride, f_shape, ceil = false)
            r = ((w / stride - 1) * stride - w + f_shape)
            if ceil
              r.odd? ? r / 2 + 1 : r / 2
            else
              r / 2
            end
          end
        end
      end
    end
  end
end