module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module VariableOps
      def VariableOps.included(klass)
        klass.class_eval do
          register_op :assign do |context, tensor, inputs|
            assign_var_buffer(tensor, inputs[0])
          end

          register_op :assign_add do |context, tensor, inputs|
            assign_op('assign_add', tensor, inputs)
          end

          register_op :assign_sub do |context, tensor, inputs|
            assign_op('assign_sub', tensor, inputs)
          end

          register_op %i[variable variable_v2], noop: true do |_context, tensor, _inputs|
            assign = tensor.inputs[0] || tensor

            read_var(assign)
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
            value = assign.inputs[0]
            buffer = complete_eval(value, child_context)
            manager = TensorStream::OpenclStorageManager.current_storage_manager

            raise "not a variable #{assign.name}" unless assign.options.key?(:var_name)

            manager.cl_assign_var(tensor.graph, _opencl_queue, assign.options[:var_name], buffer)
          end

          def assign_var_buffer(tensor, buffer)
            raise "not a variable #{tensor.name}" unless tensor.options.key?(:var_name)

            manager = TensorStream::OpenclStorageManager.current_storage_manager
            manager.cl_assign_var(tensor.graph, _opencl_queue, tensor.options[:var_name], buffer)
          end

          def read_var(tensor)
            raise "not a variable #{tensor.name}" unless tensor.options.key?(:var_name)

            manager = TensorStream::OpenclStorageManager.current_storage_manager
            manager.cl_read_var(tensor.graph, _opencl_queue, tensor.options[:var_name])
          end

          def assign_op(program, tensor, inputs)
            output_buffer = read_var(tensor)

            dtype = output_buffer.data_type
            event_wait_list = build_event_wait_list(inputs)
            var_shape = output_buffer.shape.reduce(:*) || 1
            input_shape = inputs[0].shape.reduce(:*) || 1
            mode = 0
            work_group = [ var_shape ]

            cl_row = OpenCL::Int1.new(input_shape)

            if var_shape > input_shape
              mode = 1
              raise TensorStream::ValueError, "Incompatible shape sizes #{var_shape} != #{input_shape}" if var_shape & input_shape != 0
              work_group = [var_shape / input_shape, input_shape]
            end

            output_buffer.op = _cl_program(program, dtype: tensor.data_type)
              .send("#{program}_#{dtype}_#{mode}", _opencl_queue, work_group, cl_row, output_buffer.cl_buffer,
              inputs[0].cl_buffer, event_wait_list: event_wait_list)

            output_buffer.dirty = true
            output_buffer
          end
        end
      end
    end
  end
end