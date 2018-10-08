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
            target_var, accum, lr, grad = inputs

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
            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)
            event = _cl_program('softmax_grad', dtype: dtype, size: n).
                        send(:"softmax_grad_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer,
                             grad.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end
        end
      end
    end
  end
end