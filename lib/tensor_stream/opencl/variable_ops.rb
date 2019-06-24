module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module VariableOps
      def VariableOps.included(klass)
        klass.class_eval do
          register_op :assign, noop: true do |context, tensor, inputs|
            assign_var(tensor, inputs[1], context)
          end

          register_op :assign_add do |context, tensor, inputs|
            value = execute_2_operand_func('add', tensor, inputs[0], inputs[1])
            assign_var(tensor, value, context)
          end

          register_op :assign_sub do |context, tensor, inputs|
            value = execute_2_operand_func('sub', tensor, inputs[0], inputs[1])
            assign_var(tensor, value, context)
          end

          register_op %i[variable variable_v2], noop: true do |_context, tensor, _inputs|
            assign = tensor.inputs[0] || tensor

            if assign.container_buffer.nil?
              value = assign.container

              raise "Variable #{tensor.name} not initialized!" if value.nil?

              assign.options[:container].buffer = convert_to_opencl(value, shape_eval(value), data_type: tensor.data_type, name: assign.name)
              assign.options[:container].value = value
            end
            assign.container_buffer
          end

          register_op :restore_ts do |context, tensor, inputs|
            inputs = inputs.dup
            filename = inputs.shift
            tensor_names = inputs

            filename = read_final_result(complete_eval(filename, context))
            tensor_names.map! { |n| read_final_result(complete_eval(n, context)) }

            input_dump = YAML.safe_load(File.read(filename), [Symbol])
            vars = tensor.graph.get_collection(GraphKeys::GLOBAL_VARIABLES)

            vars.select! { |v| input_dump['variables'].key?(v.name) && tensor_names.include?(v.name) }
            vars.each do |variable|
              data = TensorStream::Packer.unpack(Zlib::Inflate.inflate(Base64.decode64(input_dump['variables'][variable.name]['data'])), variable.data_type)
              shape = input_dump['variables'][variable.name]['shape']
              variable.buffer = convert_to_opencl(data, shape, data_type: variable.data_type, name: variable.name)
              variable.value = TensorShape.reshape(data, shape)
            end

            nil
          end


          protected

          def assign_var(tensor, b, child_context)
            assign = tensor.inputs[0] || tensor
            buffer = complete_eval(b, child_context)

            if assign.container_buffer
              event_wait_list = build_event_wait_list([buffer, assign.container_buffer])
            else
              var_buffer = _create_result_buffer(buffer.data_type, buffer.shape, tensor.name)
              assign.options[:container].buffer = var_buffer
            end

            assign.container_buffer.op = if assign.container_buffer.cl_buffer != buffer.cl_buffer
              _opencl_queue.enqueue_copy_buffer(buffer.cl_buffer, assign.container_buffer.cl_buffer, event_wait_list: event_wait_list)
            else
              buffer.op
            end

            assign.container_buffer.dirty = true
            assign.container_buffer
          end
        end
      end
    end
  end
end