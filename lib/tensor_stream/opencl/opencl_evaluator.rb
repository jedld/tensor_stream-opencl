require 'tensor_stream/evaluator/operation_helpers/random_gaussian'
require 'tensor_stream/evaluator/operation_helpers/array_ops_helper'
require 'tensor_stream/evaluator/operation_helpers/math_helper'
require 'tensor_stream/evaluator/buffer'
require 'tensor_stream/opencl/opencl_buffer'
require 'tensor_stream/opencl/opencl_template_helper'
require 'tensor_stream/device'
require 'tensor_stream/opencl/opencl_device'
require 'opencl_ruby_ffi'
require 'narray_ffi'
require 'tensor_stream/evaluator/base_evaluator'
require 'tensor_stream/opencl/math_ops'
require 'tensor_stream/opencl/nn_ops'
require 'tensor_stream/opencl/images_ops'
require 'tensor_stream/opencl/array_ops'
require 'tensor_stream/helpers/op_helper'

module TensorStream
  module Evaluator
    class FullEvalNotPossible < RuntimeError
    end

    # Errors during graph evaluation
    class EvaluatorExcecutionException < RuntimeError
      attr_reader :tensor

      def initialize(exception, tensor)
        @exception = exception
        @tensor = tensor
      end

      def wrapped_exception
        @exception
      end
    end

    ##
    # PURE ruby evaluator used for testing and development
    class OpenclEvaluator < BaseEvaluator
      attr_accessor :retain
      attr_reader :opencl_device

      include TensorStream::OpHelper
      include TensorStream::ArrayOpsHelper
      include TensorStream::MathHelper
      include TensorStream::OpenCLHelpers::MathOps
      include TensorStream::OpenCLHelpers::NNOps
      include TensorStream::OpenCLHelpers::ImagesOps
      include TensorStream::OpenCLHelpers::ArrayOps

      def initialize(session, device, thread_pool: nil, log_intermediates: false)
        super
        _create_opencl_context(device.native_device)
        @opencl_device = device.native_device
        create_command_queue
      end

      def self.query_supported_devices
        devices = query_devices_with_score
        devices.sort { |a| a[1] }.reverse.map do |d|
          opencl_to_device(d)
        end
      end

      def self.fetch_device(query = [])
        devices = query_devices_with_score
        platform_devices = devices.select { |d| d[0].platform.to_s.tr(' ', '_').downcase =~ /#{query[0].downcase}/ }
        opencl_to_device(platform_devices[[query[1].to_i, platform_devices.size - 1].min])
      end

      def self.opencl_to_device(d)
        device = d[0]
        index = d[3]
        platform_name = device.platform.name.tr(' ', '_').downcase
        uri = [platform_name, index].join(':')

        device_type = device.type.to_s == 'GPU' ? :gpu : :cpu

        OpenclDevice.new(uri, device_type, self).tap do |devide|
          devide.native_device = device
        end
      end

      ##
      # Select the best device available in the system for this evaluator
      def self.default_device
        devices = OpenclEvaluator.query_devices_with_score
        device = devices.sort { |a| a[1] }.reverse.first
        opencl_to_device(device)
      end

      # opencl evaluator main entrypoint
      def run(tensor, execution_context)
         result = complete_eval(tensor, execution_context)
        _opencl_queue.finish
        read_final_result(result)
      end

      def run_with_buffer(tensor, context, execution_context)
        @context = context
        @context[:_cache][:_cl_buffers] ||= {} if context[:_cache]

        if tensor.is_a?(Array)
          tensor.collect do |t|
            value = run(t, execution_context)
            Buffer.new(data_type: t.data_type, buffer: value)
          end
        else
          value = run(tensor, execution_context)
          Buffer.new(data_type: tensor.data_type, buffer: value)
        end
      end

      # buffer comes from non-opencl evaluator
      def convert_from_buffer(tensor, result)
        if result.buffer.is_a?(TensorStream::Evaluator::OutputGroup)
          converted_outputs = result.buffer.outputs.zip(result.buffer.data_types).map { |output, data_type| convert_to_opencl([output].flatten, shape_eval(output), data_type: data_type, name: tensor.name) }
          TensorStream::Evaluator::OutputGroup.new(converted_outputs, result.buffer.data_types)
        else
          convert_to_opencl([result.buffer].flatten, shape_eval(result.buffer), data_type: result.data_type, name: tensor.name)
        end
      end

      def enqueue_buffer_read(tensor, context)
        buffer = _run(tensor, context)
        if buffer.is_a?(Array)
          buffer.collect do |b|
            next b if b.buffer.size.zero?
            b.op = _opencl_queue.enqueue_read_buffer(b.cl_buffer, b.buffer, event_wait_list: build_event_wait_list([b]))
            b
          end
        else
          return buffer.outputs[0] if buffer.is_a?(OutputGroup)
          return buffer if buffer.nil?
          return [] if buffer.buffer.nil?
          return buffer if buffer.buffer.size.zero?
          buffer.op = _opencl_queue.enqueue_read_buffer(buffer.cl_buffer, buffer.buffer, event_wait_list: build_event_wait_list([buffer]))
          buffer
        end
      end

      def complete_eval(tensor, context)
        buffer = enqueue_buffer_read(tensor, context)
        events = build_event_wait_list([buffer])

        OpenCL.wait_for_events(events) unless events.empty?
        buffer
      end

      def self.query_devices_with_score
        OpenCL.platforms.flat_map do |p|
          p.devices.select { |d| d.available > 0 }.each_with_index.collect do |d, index|
            score = 0
            if d.type.to_s == 'CPU'
              score += 1
            elsif d.type.to_s == 'GPU'
              score += 4
            end

            score += 1000 if d.platform.name == 'NVIDIA CUDA'

            score += d.max_compute_units
            score += d.max_clock_frequency

            [d, score, p.name, index]
          end
        end
      end

      protected

      def prepare_input(tensor, context, options = {})
        return nil unless tensor

        tensor = resolve_placeholder(tensor)
        if options[:noop]
          tensor
        elsif options[:buffer]
          complete_eval(tensor, context)
        elsif options[:complete]
          read_final_result(complete_eval(tensor, context))
        else
          _run(tensor, context)
        end
      end

      # read result from opencl and convert to ruby
      def read_final_result(buffer)
        return buffer.map { |b| read_final_result(b) } if buffer.is_a?(Array)
        return nil if buffer.nil?

        buffer.to_ruby
      end

      def _create_opencl_context(opencl_device)
        @opencl_context = OpenCL.create_context(opencl_device)
      end

      def create_command_queue
        supported_proprties = opencl_device.queue_properties.names

        properties = []
        properties << OpenCL::CommandQueue::PROFILING_ENABLE if supported_proprties.include?('PROFILING_ENABLE')
        properties << OpenCL::CommandQueue::OUT_OF_ORDER_EXEC_MODE_ENABLE if supported_proprties.include?('OUT_OF_ORDER_EXEC_MODE_ENABLE')
        @command_queue = _opencl_context.create_command_queue(opencl_device, properties: properties)
      end

      def _opencl_context
        @opencl_context
      end

      def _opencl_queue
        @command_queue
      end

      def cl_template_path(kernel, extension)
        File.join(File.dirname(__FILE__), 'kernels', "#{kernel}.#{extension}")
      end

      def _cl_program(kernel, args = {})
        suffix = args.collect { |k, v| "#{k}.#{escape_arg_content(v)}" }.join('.')
        @context[:_cache]["_opencl_kernel_#{kernel}.#{suffix}:#{object_id}"] ||= begin
          file_path = File.join('/tmp', "#{kernel}.#{suffix}.cl")
          source = if File.exist?(file_path)
                     File.read(file_path)
                   else
                     filename = %w[cl.erb cl].map { |ext| cl_template_path(kernel, ext) }.find { |n| File.exist?(n) }
                     raise "opencl kernel template for #{kernel} has not yet been defined" if filename.nil?
                     source = File.read(filename)
                     source = OpenclTemplateHelper.new(source).generate(args)
                     File.write(file_path, source)
                     source
                   end
          program = _opencl_context.create_program_with_source(source)
          program.build
        rescue OpenCL::Error::BUILD_PROGRAM_FAILURE => e
          puts "OpenCL Compile error: #{program.build_log}"
          raise e
        end
      end

      def escape_arg_content(value)
        return value.tr(' ','_') if value.is_a?(String)
        return value.join('-') if value.is_a?(Array)

        value
      end

      def _run(tensor, execution_context)
        return tensor if tensor.is_a?(OpenCLBuffer)
        return tensor.map { |t| _run(t, execution_context) } if tensor.is_a?(Array) && !tensor.size.empty? && tensor[0].is_a?(Tensor)

        tensor = tensor.call if tensor.is_a?(Proc)

        child_context = execution_context.dup
        res = if tensor.is_a?(Operation)
                if !self.class.ops.include?(tensor.operation.to_sym)
                  result = @session.delegate_to_evaluator(tensor, @context, execution_context)
                  convert_from_buffer(tensor, result)
                else
                  eval_operation(tensor, child_context)
                end
              elsif tensor.is_a?(Variable)
                eval_variable(tensor, child_context)
              elsif tensor.is_a?(Placeholder)
                resolve_placeholder(tensor, child_context)
              else
                eval_tensor(tensor, child_context)
              end
        execution_context.deep_merge!(returns: child_context[:returns])
        res
      end

      def eval_variable(tensor, _child_context)
        raise "variable #{tensor.name} not initalized" if tensor.value.nil? && (tensor.buffer.nil? || !tensor.buffer.dirty)
        tensor.buffer = wrap_opencl(tensor, name: tensor.name) if tensor.buffer.nil?
        tensor.buffer
      end

      register_op :no_op do |_context, _tensor, _inputs|
      end

      register_op :cond, noop: true do |context, tensor, inputs|
        pred = complete_eval(tensor.options[:pred], context)

        if all_true?(pred.buffer)
          complete_eval(inputs[0], context)
        else
          complete_eval(inputs[1], context)
        end
      end

      register_op :identity do |context, tensor, inputs|
        if tensor.inputs.size > 1
          tensor.inputs[1..inputs.size].each { |input| complete_eval(input, context) }
        end
        inputs[0]
      end

      register_op :assign, noop: true do |context, tensor, inputs|
        assign_var(tensor, inputs[1], context)
      end

      register_op :assign_add do |context, tensor, inputs|
        value = execute_2_operand_func('add', tensor, inputs[0], inputs[1], context)
        assign_var(tensor, value, context)
      end

      register_op :assign_sub do |context, tensor, inputs|
        value = execute_2_operand_func('sub', tensor, inputs[0], inputs[1], context)
        assign_var(tensor, value, context)
      end

      register_op :variable, noop: true do |context, tensor, inputs|
        variable = tensor.inputs[0]
        raise "variable #{tensor.name} not initalized" if variable.value.nil? && (variable.buffer.nil? || !variable.buffer.dirty)
        variable.buffer = wrap_opencl(variable, name: variable.name) if variable.buffer.nil?
        variable.buffer
      end

      %i[less less_equal greater greater_equal equal not_equal logical_and].each do |op|
        register_op op, noop: true do |context, tensor, inputs|
          execute_2_operand_func(op.to_s, tensor, inputs[0], inputs[1], context, 'cond')
        end
      end

      register_op :where, noop: true do |context, tensor, inputs|
        pred = tensor.options[:pred]
        execute_cond_func('where', tensor, pred, inputs[0], inputs[1], context)
      end

      register_op :check_numerics, noop: true do |context, tensor, inputs|
        a = complete_eval(inputs[0], context)
        name = tensor.options[:name]

        a.buffer.each do |input|
          raise TensorStream::InvalidArgumentError, "#{name} Invalid Argument" if input.nan? || input.infinite?
        end
        a
      end

      register_op :broadcast_transform do |context, tensor, inputs|
        a, b = inputs

        if a.shape == b.shape
          [a, b]
        else
          input_a = read_final_result(complete_eval(a, context))
          input_b = read_final_result(complete_eval(b, context))
          b_a, b_b = broadcast(input_a, input_b)
          [wrap_opencl(b_a, data_type: a.data_type, name: "#{tensor.name}_a"),
           wrap_opencl(b_b, data_type: a.data_type, name: "#{tensor.name}_b")]
        end
      end

      register_op :print do |context, tensor, inputs|
        a, b = inputs
        input_b = complete_eval(b, context)
        input_b = read_final_result(input_b)
        puts "#{tensor.options.fetch(:message, '')} #{input_b}"
        a
      end

      register_op :stop_gradient do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op :broadcast_gradient_args, buffer: true do |_context, tensor, inputs|
        rx, ry = get_broadcast_gradient_args(inputs[0].buffer.to_a, inputs[1].buffer.to_a)
        OutputGroup.new([wrap_opencl(rx, data_type: :int32, name: tensor.name), wrap_opencl(ry, data_type: :int32, name: "#{tensor.name}:1")], tensor.inputs.map(&:data_type))
      end

      register_op :flow_group do |_context, _tensor, inputs|
        events = build_event_wait_list(inputs)
        OpenCL.wait_for_events(events) unless events.empty?
        nil
      end

      register_op :size do |_context, tensor, inputs|
        wrap_opencl(inputs[0].buffer.size, name: tensor.name, data_type: tensor.options[:out_type] || :int32)
      end

      def eval_operation(tensor, child_context)
        cache_key = "#{tensor.graph.object_id}_opencl_#{tensor.name}:#{object_id}"
        return @context[:_cache][cache_key] if @context[:_cache].key?(cache_key)
        return @context[cache_key] if @context.key?(cache_key)
        # puts "opencl: #{tensor.name}"
        invoke(tensor, child_context).tap do |result|
          if tensor.breakpoint
            a = resolve_placeholder(tensor.inputs[0], child_context) if tensor.inputs && tensor.inputs[0]
            b = resolve_placeholder(tensor.inputs[1], child_context) if tensor.inputs && tensor.inputs[1]
            a = read_final_result(complete_eval(a, child_context))
            b = read_final_result(complete_eval(b, child_context))
            result = read_final_result(complete_eval(result, child_context))

            tensor.breakpoint.call(tensor, a, b, result)
          end
          if @log_intermediates
            @context[:compute_history] << {
              name: tensor.name,
              type: tensor.data_type,
              shape: shape_eval(result),
              source: tensor.source,
              description: tensor.to_math(true, 1),
              value: result
            }
          end
          @context[cache_key] = result
          @context[:_cache][cache_key] = result if tensor.is_const
        end
      rescue EvaluatorExcecutionException => e
        _opencl_queue.finish # dump queue
        raise e, "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math(true, 1)} defined at #{tensor.source}"
      rescue TensorStreamError => e
        _opencl_queue.finish # dump queue
        raise e, "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math(true, 1)} defined at #{tensor.source}"
      rescue StandardError => e
        _opencl_queue.finish # dump queue
        puts e.message
        puts e.backtrace.join("\n")

        # shape_a = a.shape.shape if a
        # shape_b = b.shape.shape if b
        # dtype_a = a.data_type if a
        # dtype_b = b.data_type if b
        # a = complete_eval(a, child_context)
        # b = complete_eval(b, child_context)
        # puts "name: #{tensor.given_name}"
        # # puts "op: #{tensor.to_math(true, 1)}"
        # puts "A #{shape_a} #{dtype_a}: #{a}" if a
        # puts "B #{shape_b} #{dtype_b}: #{b}" if b
        # dump_intermediates if @log_intermediates
        # File.write('/home/jedld/workspace/tensor_stream/samples/error.graphml', TensorStream::Graphml.new.get_string(tensor, @session))

        # File.write('/Users/josephemmanueldayo/workspace/gradients.graphml', TensorStream::Graphml.new.get_string(tensor, @session))
        raise EvaluatorExcecutionException.new(e, tensor), "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math(true, 1)} defined at #{tensor.source}"
      end

      def eval_tensor(tensor, child_context)
        return tensor unless tensor.is_a?(Tensor)

        cache_key = "#{tensor.graph.object_id}_opencl_#{tensor.name}:#{object_id}"
        return @context[cache_key] if @context.key?(cache_key)
        return @context[:_cache][cache_key] if tensor.is_const && @context[:_cache][cache_key]
        @context[cache_key] = if tensor.value.is_a?(Tensor)
                                _run(tensor.value, child_context)
                              else
                                wrap_opencl(tensor, name: tensor.name)
                              end
        @context[:_cache][cache_key] = @context[cache_key] if tensor.is_const
        @context[cache_key]
      end

      private

      def assign_var(tensor, b, child_context)
        assign = tensor.inputs[0] || tensor
        buffer = complete_eval(b, child_context)

        if assign.buffer
          # buffer = type_cast(buffer, assign.data_type, name: "#{tensor.name}/cast_#{tensor.name}_#{tensor.data_type}")
          event_wait_list = build_event_wait_list([buffer, assign.buffer])
          assign.buffer.op = if assign.buffer.cl_buffer != buffer.cl_buffer
                               _opencl_queue.enqueue_copy_buffer(buffer.cl_buffer, assign.buffer.cl_buffer, event_wait_list: event_wait_list)
                             else
                               buffer.op
                             end
        else
          value = read_final_result(buffer)
          assign.buffer = convert_to_opencl(value, buffer.shape, data_type: tensor.data_type, name: assign.name)
          assign.value = value
        end
        assign.buffer.dirty = true
        assign.buffer
      end

      def execute_2_operand_func(op_name, tensor, input_a, input_b, child_context, prog_name = nil)
        a = _run(input_a, child_context)
        b = _run(input_b, child_context)
        a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
        dtype = tensor.data_type
        result_shape = TensorShape.infer_shape(a.shape, b.shape)
        return _create_result_buffer(dtype, [0], "out_#{tensor.name}") if result_shape == [0]
        output_buffer = _create_result_buffer(tensor.data_type, result_shape, "out_#{tensor.name}")
        a, b, prog, switch_operands = select_program(a, b, op_name)
        m, n = result_shape
        work_group = [m || 1, n || 1]
        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)
        cl_switch = OpenCL::Int1.new(switch_operands) # no need to switch for addition

        event_wait_list = build_event_wait_list([a, b]) # add dependency wait list

        method_call = :"#{prog}_#{a.data_type}_#{b.data_type}"
        prog_name ||= op_name
        event = if prog == "#{op_name}_b"
                  cl_m_b, cl_n_b = if b.shape.size == 2
                                     [OpenCL::Int1.new(b.shape[0]), OpenCL::Int1.new(b.shape[1])]
                                   elsif b.shape.size == 1
                                     [OpenCL::Int1.new(1), OpenCL::Int1.new(b.shape[0])]
                                   else
                                     raise "rank > 2 not supported!"
                                   end
                  _cl_program(prog_name, a: a.data_type, b: b.data_type, dtype: dtype).
                    send(method_call, _opencl_queue, work_group, cl_m, cl_n, cl_m_b, cl_n_b,
                         cl_switch, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
                else
                  _cl_program(prog_name, a: a.data_type, b: b.data_type, dtype: dtype).
                    send(method_call, _opencl_queue, work_group, cl_m, cl_n, cl_switch,
                         a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
                end

        output_buffer.op = event
        output_buffer
      end

      def execute_cond_func(op_name, tensor, pred, input_a, input_b, child_context)
        p = _run(pred, child_context)
        a = _run(input_a, child_context)
        b = _run(input_b, child_context)

        a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
        dtype = tensor.data_type

        output_buffer = _create_result_buffer(tensor.data_type, p.shape, tensor.name)

        m, n = p.shape
        work_group = [m || 1, n || 1]
        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        event_wait_list = build_event_wait_list([a, b, p]) # add dependency wait list
        output_buffer.op = _cl_program(op_name.to_s, dtype: dtype).send(:"#{op_name}_#{dtype}", _opencl_queue, work_group, cl_m, cl_n, p.cl_buffer, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        output_buffer
      end

      def execute_func(op_name, tensor, a, child_context)
        a = _run(a, child_context)
        event_wait_list = build_event_wait_list([a])
        dtype = tensor.data_type
        output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

        m, n = a.shape
        work_group = [m || 1, n || 1]
        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        event = _cl_program(op_name.to_s, dtype: dtype).send(:"#{op_name}_#{dtype}", _opencl_queue, work_group, cl_m, cl_n, a.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        output_buffer.op = event
        output_buffer
      end

      def auto_type_cast(a, b, name: nil)
        return [a, b] if a.data_type == b.data_type
        m, n = b.shape
        work_group = [m || 1, n || 1]
        event_wait_list = build_event_wait_list([b])
        buffer = _create_result_buffer(b.data_type, b.shape, name)

        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        buffer.op = _cl_program("cast", source_dt: a.data_type, target_dt: b.data_type).cast(_opencl_queue, work_group, cl_m, cl_n, b.cl_buffer, buffer.cl_buffer, event_wait_list: event_wait_list)
        [a, buffer]
      end

      def type_cast(source, data_type, name: nil)
        return source if source.data_type == data_type
        m, n = source.shape
        work_group = [m || 1, n || 1]
        event_wait_list = [source.op].compact
        buffer = _create_result_buffer(data_type, source.shape, name)

        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        buffer.op = _cl_program("cast", source_dt: source.data_type, target_dt: data_type).cast(_opencl_queue, work_group, cl_m, cl_n, source.cl_buffer, buffer.cl_buffer, event_wait_list: event_wait_list)
        buffer
      end

      def wrap_opencl(tensor, data_type: nil, name: nil)
        value, shape = if tensor.is_a?(Tensor)
                         [tensor.value, tensor.shape.shape]
                       else
                         [tensor, shape_eval(tensor)]
                       end

        convert_to_opencl(value, shape, data_type: data_type || tensor.data_type, name: name)
      end

      def get_cached_buffer(name, shape)
        cache_key = "_cl_object_#{name}:#{shape.join('_')}:#{object_id}"
        @context[:_cache][cache_key]
      end

      def convert_to_opencl(value, shape, data_type: nil, name: nil)
        value = [value] if !value.is_a?(Array) && !value.is_a?(NArray)

        cache_key = "_cl_object_#{name}:#{shape.join('_')}:#{object_id}"
        cl_object = if name && @context[:_cache][cache_key]
                      @context[:_cache][cache_key]
                    else
                      narray_size = shape.reduce(:*) || 1
                      cl_buffer_size = shape.empty? ? 1 : shape.reduce(:*)

                      buffer = if value.is_a?(NArray)
                                 value
                               elsif data_type == :string && shape.empty?
                                 cl_buffer_size = value[0].bytesize
                                 allocate_narray_for_type(data_type, value[0].bytesize)
                               else
                                 allocate_narray_for_type(data_type, narray_size)
                               end

                      return nil if buffer.nil?



                      cl_buffer = unless value.flatten.empty?
                                    cl_buffer_size = 1 if cl_buffer_size.zero?
                                    _opencl_context.create_buffer(cl_buffer_size * buffer.element_size)
                                  end

                      @context[:_cache][cache_key] = OpenCLBuffer.new(name: name, data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer)
                    end
        if data_type == :string
          value[0].each_byte.with_index do |c, index|
            cl_object.buffer[index] = c
          end
        elsif value.is_a?(Array)
          value.flatten.each_with_index do |element, index|
            cl_object.buffer[index] = if element.is_a?(Tensor)
                                        read_final_result(complete_eval(element, {}))
                                      elsif data_type == :boolean
                                        element ? 1 : 0
                                      else
                                        Tensor.cast_dtype(element, data_type)
                                      end
          end
        elsif value.is_a?(NArray)
          cl_object.buffer = value
        elsif data_type == :boolean
          cl_object.buffer[0] = element ? 1 : 0
        else
          cl_object.buffer[0] = Tensor.cast_dtype(value, data_type)
        end

        if cl_object.cl_buffer && !value.nil? && (!value.is_a?(Array) || !value.empty?)
          write_op = _opencl_queue.enqueue_write_buffer(cl_object.cl_buffer, cl_object.buffer)
        end

        cl_object.op = write_op
        cl_object
      end

      def allocate_narray_for_type(data_type, narray_size)
        case data_type
        when :float, :float32
          NArray.sfloat(narray_size)
        when :float64
          NArray.float(narray_size)
        when :int, :int32, :int64
          NArray.int(narray_size)
        when :int16
          NArray.sint(narray_size)
        when :uint8
          NArray.byte(narray_size)
        when :boolean
          NArray.byte(narray_size)
        when :string
          NArray.byte(narray_size)
        when :unknown
          nil
        else
          raise "unsupported type #{data_type}"
        end
      end

      def _create_result_buffer(data_type, shape, name)
        return OpenCLBuffer.new(name: name, data_type: data_type, shape: [0], buffer: nil, cl_buffer: nil) if shape == [0]
        @context[:_cache][:_cl_buffers]["_result_#{name}_#{shape.join('_')}:#{object_id}"] ||= begin
          size = shape.empty? || shape == [0] ? 1 : shape.reduce(:*)
          buffer =  allocate_narray_for_type(data_type, size)
          cl_buffer = _opencl_context.create_buffer(buffer.size * buffer.element_size)
          OpenCLBuffer.new(data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer, name: name)
        end
      end

      # automatically use sub buffers
      def _create_result_sub_buffer(parent_buffer, index, data_type, shape, name)
        cache_key ="_sub_result_#{parent_buffer.object_id}_#{name}_#{index}:#{object_id}"
        @context[:_cache][:_cl_buffers][cache_key] ||= begin
          size = shape.empty? || shape == [0] ? 1 : shape.reduce(:*)
          buffer = allocate_narray_for_type(data_type, size)

          if parent_buffer.cl_buffer.associated_memobject.nil?
            start = index * buffer.size * buffer.element_size
            region = OpenCL::BufferRegion::new(start, buffer.size * buffer.element_size)
            cl_buffer = parent_buffer.cl_buffer.create_sub_buffer(OpenCL::BUFFER_CREATE_TYPE_REGION, region)
            OpenCLBuffer.new(data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer, name: name)
          else
            _create_result_buffer(tensor.data_type, shape, name)
          end
        end

        buffer = @context[:_cache][:_cl_buffers][cache_key]

        if buffer.cl_buffer.associated_memobject
          buffer.op = parent_buffer.op
        else
          region_size_in_bytes = buffer.buffer.size * buffer.buffer.element_size
          start = index * region_size_in_bytes
          region = [region_size_in_bytes, 1, 1]
          buffer.op = _opencl_queue.enqueue_copy_buffer_rect(parent_buffer.cl_buffer, buffer.cl_buffer, region, src_origin: [start, 0, 0], event_wait_list: parent_buffer.op)
        end

        buffer
      end

      def get_op_with_axis(a, target_axis, current_axis, output_type, op = ->(t, u) { t > u })
        if target_axis == current_axis
          if a[0].is_a?(Array)
            (0...a[0].size).each.collect do |column_index|
              max = nil
              max_index = 0
              a.each_with_index do |row, row_index|
                if max.nil? || op.call(row[column_index], max)
                  max = row[column_index]
                  max_index = row_index
                end
              end

              Tensor.cast_dtype(max_index, output_type)
            end
          else
            max = nil
            max_index = 0
            a.each_with_index do |x, index|
              if max.nil? || op.call(x, max)
                max = x
                max_index = index
              end
            end
            Tensor.cast_dtype(max_index, output_type)
          end
        else
          a.collect do |row|
            get_op_with_axis(row, target_axis, current_axis + 1, output_type, op)
          end
        end
      end

      def _reduced_shape(input_shape, axes)
        return [] if axes.nil? # reduce to scalar
        axes = [axes] unless axes.is_a?(Array)
        return input_shape if axes.empty?

        axes.each do |dimen|
          input_shape[dimen] = 1
        end
        input_shape
      end

      def reduction(child_context, tensor, a, b, func)
        input = complete_eval(a, child_context)
        axis = b.is_a?(Tensor) ? read_final_result(complete_eval(b, child_context)) : b
        if axis.nil?
          red = input.buffer.send(func)
          convert_to_opencl(red, [], data_type: tensor.data_type, name: tensor.name)
        else
          return input if input.shape.empty?
          value = input.buffer.reshape(*input.shape.reverse)
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

      # selects variants of cl programs depending on input
      def select_program(input_a, input_b, op)
        return [input_a, input_b, op.to_s, 0] if input_a.shape == input_b.shape

        return [input_b, input_a, "#{op}_c", 1] if input_a.shape.empty? || input_a.shape.reduce(:*) == 1 # A is scalar?
        return [input_a, input_b, "#{op}_c", 0] if input_b.shape.empty? || input_a.shape.reduce(:*) == 1 # B is scalar?

        return [input_b, input_a, "#{op}_b", 1] if input_a.shape.size < input_b.shape.size

        if input_a.shape.size == input_b.shape.size
          input_a.shape.zip(input_b.shape).each do |s1, s2|
            return [input_b, input_a, "#{op}_b", 1] if s1 < s2
          end
        end

        [input_a, input_b, "#{op}_b", 0]
      end

      def _rank_from_shape(shape)
        shape.is_a?(Array) ? shape.size : 0
      end

      def build_event_wait_list(inputs)
        inputs.flatten.compact.map(&:op).compact
      end

      def resolve_placeholder(placeholder, _execution_context = {})
        return nil if placeholder.nil?

        var = if placeholder.is_a?(Placeholder)
                @context[placeholder.name.to_sym].tap do |c|
                  raise "missing placeholder #{placeholder.name}" if c.nil?
                end
              else
                placeholder
              end

        return convert_to_opencl(var, shape_eval(var), data_type: placeholder.data_type, name: placeholder.name) unless var.is_a?(Tensor)
        Tensor.cast_dtype(var, placeholder.data_type)
      end

      def all_true?(arr)
        if arr.is_a?(Array) || arr.is_a?(NArray)
          arr.each do |a|
            return false unless all_true?(a)
          end
          return true
        end

        arr != 0
      end
    end
  end
end

TensorStream::Evaluator.register_evaluator(TensorStream::Evaluator::OpenclEvaluator, 'opencl', 1)
