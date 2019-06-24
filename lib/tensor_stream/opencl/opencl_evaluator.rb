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
require 'tensor_stream/opencl/random_ops'
require 'tensor_stream/opencl/variable_ops'
require 'tensor_stream/helpers/op_helper'
require 'tensor_stream/opencl/opencl_storage_manager'

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
    # OpenCL hardware accelerated evaluator
    #
    class OpenclEvaluator < BaseEvaluator
      attr_accessor :retain
      attr_reader :opencl_device, :opencl_context
      attr_writer :context

      include TensorStream::OpHelper
      include TensorStream::ArrayOpsHelper
      include TensorStream::MathHelper
      include TensorStream::OpenCLHelpers::MathOps
      include TensorStream::OpenCLHelpers::NNOps
      include TensorStream::OpenCLHelpers::ImagesOps
      include TensorStream::OpenCLHelpers::ArrayOps
      include TensorStream::OpenCLHelpers::RandomOps
      include TensorStream::OpenCLHelpers::VariableOps
      include TensorStream::CLEventHelpers

      def initialize(session, device, thread_pool: nil, log_intermediates: false)
        super
        _create_opencl_context
        @opencl_device = device.native_device

        @max_work_item_dimensions = @opencl_device.max_work_item_dimensions
        @max_work_item_sizes = @opencl_device.max_work_item_sizes
        @max_work_group_size = @opencl_device.max_work_group_size

        @local_mem_size = @opencl_device.local_mem_size
        @device_type = @opencl_device.type.to_s.downcase

        create_command_queue
      end

      class << self
        def get_storage_manager
          OpenclStorageManager.current_storage_manager
        end

        def query_supported_devices
          devices = query_devices_with_score
          devices.sort_by { |a| a[1] }.map do |d|
            opencl_to_device(d)
          end
        end

        def fetch_device(query = [])
          devices = query_devices_with_score
          platform_devices = devices.select { |d| d[0].platform.to_s.tr(' ', '_').downcase =~ /#{query[0].downcase}/ }
          opencl_to_device(platform_devices[[query[1].to_i, platform_devices.size - 1].min])
        end

        def opencl_to_device(dev)
          device = dev[0]
          index = dev[3]
          platform_name = device.platform.name.tr(' ', '_').downcase
          uri = [platform_name, index].join(':')

          device_type = device.type.to_s == 'GPU' ? :gpu : :cpu

          OpenclDevice.new(uri, device_type, self).tap do |d|
            d.native_device = device
          end
        end

        ##
        # Select the best device available in the system for this evaluator
        def default_device
          devices = OpenclEvaluator.query_devices_with_score
          device = devices.max { |a, b| a[1] <=> b[1] }
          opencl_to_device(device)
        end

        def getset_global_opencl_context(platform)
          @global_opencl_context ||= {}
          @global_opencl_context[platform] ||= yield
          @global_opencl_context[platform]
        end
      end

      # opencl evaluator main entrypoint
      def run(tensor, execution_context)
        result = complete_eval(tensor, execution_context)
        # puts "-------------------wait finish------------------------"
        _opencl_queue.finish
        # puts "-------------------done finish------------------------"
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
          converted_outputs = result.buffer.outputs.zip(result.buffer.data_types).map do |output, data_type|
            convert_to_opencl([output].flatten, shape_eval(output), data_type: data_type, name: tensor.name)
          end
          TensorStream::Evaluator::OutputGroup.new(converted_outputs, result.buffer.data_types)
        else
          convert_to_opencl([result.buffer].flatten, shape_eval(result.buffer), data_type: result.data_type, name: tensor.name)
        end
      end

      # Generate OpenCL instruction to read back from GPU memory to Host memory for a tensor
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

          # lazy allocate
          buffer.buffer = OpenCLBuffer.allocate_narray_for_type(buffer.buffer.data_type, buffer.buffer.size) if buffer.buffer.is_a?(OpenCLBuffer::LazyBuffer)

          buffer.op = _opencl_queue.enqueue_read_buffer(buffer.cl_buffer, buffer.buffer, event_wait_list: build_event_wait_list([buffer]))
          buffer
        end
      end

      def complete_eval(tensor, context)
        return nil if tensor.nil?

        buffers = if tensor.is_a?(Array)
                    tensor.map { |t|
                      enqueue_buffer_read(t, context)
                    }
                  else
                    [enqueue_buffer_read(tensor, context)]
                  end

        events = build_event_wait_list(buffers)
        # puts "** wait #{tensor.name} **"
        OpenCL.wait_for_events(events) unless events.empty?
        # puts "** done #{tensor.name} **"
        tensor.is_a?(Array) ? buffers : buffers.first
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

            score += d.max_compute_units * d.max_clock_frequency

            [d, score, p.name, index]
          end
        end
      end

      protected

      ##
      # called when passing control to another evaluator
      def perform_transition(tensor, input, next_evaluator, execution_context)
        if next_evaluator.is_a?(OpenclEvaluator) # OpenCL but different device?
          # create opencl buffer for this tensor
          next_evaluator.context = @context

          foreign_buffer = next_evaluator._run(input, execution_context)
          event_list = build_event_wait_list([foreign_buffer])

          output_buffer = _create_result_buffer(input.data_type, foreign_buffer.shape, "t_#{tensor.name}_#{input.name}")
          output_buffer.op = if next_evaluator.opencl_context == @opencl_context
                               _opencl_queue.enqueue_copy_buffer(foreign_buffer.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_list)
                             else
                               puts "wait finish transition ** #{input.name} **"
                               read_event = next_evaluator._opencl_queue.enqueue_read_buffer(foreign_buffer.cl_buffer, output_buffer.buffer, event_wait_list: event_list)
                               OpenCL.wait_for_events(read_event)
                               _opencl_queue.enqueue_write_buffer(output_buffer.cl_buffer, output_buffer.buffer)
                             end
          output_buffer
        else
          super
        end
      end

      def prepare_input(tensor, context, options = {})
        return nil unless tensor

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

      def _create_opencl_context(device = nil)
        if device.nil?
          all_devices_by_platform = {}
          TensorStream::Evaluator::OpenclEvaluator.query_supported_devices.map(&:native_device).each do |d|
            all_devices_by_platform[d.platform.name] ||= []
            all_devices_by_platform[d.platform.name] << d
          end

          all_devices_by_platform.each do |platform, devices|
            @opencl_context = TensorStream::Evaluator::OpenclEvaluator.getset_global_opencl_context(platform) do
              OpenCL.create_context(devices)
            end
          end
        else
          puts "context created for #{device.native_device}"
          @opencl_context = TensorStream::Evaluator::OpenclEvaluator.getset_global_opencl_context(device.native_device.platform) do
            OpenCL.create_context(device.native_device)
          end
        end
      end

      def create_command_queue
        supported_proprties = opencl_device.queue_properties.names

        properties = []
        properties << OpenCL::CommandQueue::PROFILING_ENABLE if supported_proprties.include?('PROFILING_ENABLE')
        properties << OpenCL::CommandQueue::OUT_OF_ORDER_EXEC_MODE_ENABLE if supported_proprties.include?('OUT_OF_ORDER_EXEC_MODE_ENABLE')
        # puts "creating queue with properties #{supported_proprties}"
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
        kernel_cache_key = "_opencl_kernel_#{kernel}.#{suffix}:#{object_id}"
        @context[:_cache][kernel_cache_key] ||=
          begin
            # puts "building #{kernel_cache_key}"
            file_path = File.join(ENV['TS_OPENCL_FILE_CACHE_PATH'] || '/tmp', "#{kernel}.#{suffix}.cl")
            source = if File.exist?(file_path) && ENV['TS_OPENCL_FILE_CACHE']
                       File.read(file_path)
                     else
                       filenames = ['', ".#{@device_type}"].map { |type| %w[cl.erb cl].map { |ext| cl_template_path("#{kernel}#{type}", ext) } }.flatten
                       filename = filenames.find { |n| File.exist?(n) }
                       raise "opencl kernel template for #{kernel} has not yet been defined" if filename.nil?

                       source = File.read(filename)
                       source = OpenclTemplateHelper.new(source).generate(args)
                       File.write(file_path, source) if ENV['TS_OPENCL_FILE_CACHE']
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
        return value.tr(' ', '_') if value.is_a?(String)
        return value.join('-') if value.is_a?(Array)

        value
      end

      def _run(tensor, execution_context)
        return tensor if tensor.is_a?(OpenCLBuffer)
        return tensor.map { |t| _run(t, execution_context) } if tensor.is_a?(Array) && !tensor.size.empty? && tensor[0].is_a?(Tensor)

        tensor = tensor.call if tensor.is_a?(Proc)

        child_context = execution_context.dup
        res = if !on_same_device?(tensor) # tensor is on another device or evaluator
                perform_transition(tensor, tensor, @context[:_cache][:placement][tensor.name][1], execution_context)
              elsif tensor.is_a?(Operation)
                eval_operation(tensor, child_context)
              elsif tensor.is_a?(Variable)
                eval_operation(tensor.op, child_context)
              else
                raise "invalid tensor type! #{tensor.class}"
              end

        execution_context.deep_merge!(returns: child_context[:returns])
        res
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

      register_op :identity do |_context, tensor, inputs|
        value = inputs[0]
        if value.is_a?(OutputGroup)
          value
        else
          buffer = OpenCLBuffer.new(self, name: tensor.name, data_type: tensor.data_type, shape: value.shape, buffer: value.buffer, cl_buffer: value.cl_buffer)
          buffer.op = build_event_wait_list(inputs)
          buffer
        end
      end

      register_op :placeholder, noop: true do |context, tensor, _inputs|
        ph = @context[tensor.name.to_sym].tap do |c|
          raise TensorStream::ValueError, "missing placeholder #{tensor.name}" if c.nil?

          if tensor.shape.shape
            value_shape = shape_eval(c)
            placeholder_shape = tensor.shape.shape
            placeholder_shape.zip(value_shape).each do |p_shape, v_shape|
              next if p_shape.nil?
              raise TensorStream::ValueError, "placeholder expects #{placeholder_shape}, got #{value_shape}" if p_shape != v_shape
            end
          end
        end

        if ph.is_a?(Tensor)
          raise TensorStream::ValueError, "placeholder expects type #{tensor.data_type}, got #{ph.data_type}" if ph.data_type != tensor.data_type

          global_eval(tensor, ph, context)
        else
          convert_to_opencl(ph, shape_eval(ph), data_type:  tensor.data_type, name: tensor.name)
        end
      end

      %i[less less_equal greater greater_equal equal not_equal logical_and].each do |op|
        register_op op do |_context, tensor, inputs|
          execute_2_operand_func(op.to_s, tensor, inputs[0], inputs[1], 'cond')
        end
      end

      register_op :where do |context, tensor, inputs|
        pred = inputs[0]

        execute_cond_func('where', tensor, pred, inputs[1], inputs[2], context)
      end

      register_op :case, noop: true do |context, tensor, _inputs|
        pred = read_final_result(complete_eval(tensor.inputs[0], context))
        result = nil

        if tensor.options[:exclusive]
          p_true = pred.each_with_index.collect { |p, index| [p, index] }.select { |a| a[0] }
          raise TensorStream::ValueError, "more than one predicate returns true pos #{p_true.map { |a| a[1] }.join(',')}" if p_true.size > 1
        end

        pred.each_with_index do |p, index|
          next unless p

          result = _run(tensor.inputs[2 + index], context)

          break unless result.nil?
        end

        result = _run(tensor.inputs[1], context) if result.nil?
        result
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
        # puts "** wait for event flow_group**"
        OpenCL.wait_for_events(events) unless events.empty?
        # puts "** done for event flow_group**"
        nil
      end

      register_op :const do |_context, tensor, inputs|
        cache_key = "const_#{tensor.graph.object_id}_opencl_#{tensor.name}:#{object_id}"
        @context[:_cache][cache_key] ||= wrap_opencl(tensor.const_value, name: tensor.name, data_type: tensor.data_type)
      end

      register_op :size do |_context, tensor, inputs|
        wrap_opencl(inputs[0].buffer.size, name: tensor.name, data_type: tensor.options[:out_type] || :int32)
      end

      def eval_operation(tensor, child_context)
        cache_key = "#{tensor.graph.object_id}_opencl_#{tensor.name}:#{object_id}"
        return @context[:_cache][cache_key] if @context[:_cache].key?(cache_key)
        return @context[cache_key] if @context.key?(cache_key)

        # puts "opencl eval #{tensor.operation}"
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
        puts e.message
        raise e, "error #{e.message} while evaluating #{tensor.name} : defined at #{tensor.source}"
      rescue TensorStreamError => e
        _opencl_queue.finish # dump queue
        puts e.message
        raise e, "error #{e.message} while evaluating #{tensor.name} : defined at #{tensor.source}"
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
        raise EvaluatorExcecutionException.new(e, tensor), "error #{e.message} while evaluating #{tensor.name} : defined at #{tensor.source}"
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

      def execute_2_operand_func(op_name, tensor, a, b, prog_name = nil)
        a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
        dtype = tensor.data_type
        result_shape = TensorShape.infer_shape(a.shape, b.shape)
        return OpenCLBuffer.nil_buffer(self, "out_#{tensor.name}", dtype) if result_shape == [0]

        output_buffer = _create_result_buffer(tensor.data_type, result_shape, "out_#{tensor.name}")
        a, b, prog, switch_operands = select_program(a, b, op_name)
        m, n = result_shape

        work_group = if result_shape.size > 2 && (b.shape.size.zero? || (a.shape == b.shape))
                       [m, result_shape.reduce(:*) / m]
                     elsif result_shape.size <= 2
                       [m || 1, n || 1]
                     elsif (b.shape.size == 1) && (result_shape.last == b.shape.last)
                      last_dim = b.shape.last
                      [result_shape.reduce(:*) / last_dim, last_dim]
                     else
                       raise "rank > 2 not supported for now"
                     end

        cl_m = OpenCL::Int1.new(work_group[0])
        cl_n = OpenCL::Int1.new(work_group[1])

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

      def execute_cond_func(op_name, tensor, p, a, b, child_context)
        a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
        dtype = tensor.data_type

        output_buffer = _create_result_buffer(tensor.data_type, p.shape, tensor.name)

        m, n = p.shape

        work_group = if p.shape.size > 2
                       [m, p.shape.reduce(:*) / m]
                     else
                       [m || 1, n || 1]
                     end

        cl_m = OpenCL::Int1.new(work_group[0])
        cl_n = OpenCL::Int1.new(work_group[1])

        event_wait_list = build_event_wait_list([a, b, p]) # add dependency wait list
        output_buffer.op = _cl_program(op_name.to_s, dtype: dtype).
                                        send(:"#{op_name}_#{dtype}", _opencl_queue, work_group,
                                              cl_m, cl_n, p.cl_buffer, a.cl_buffer, b.cl_buffer,
                                              output_buffer.cl_buffer, event_wait_list: event_wait_list)
        output_buffer
      end

      def execute_func(op_name, tensor, a, child_context)
        a = _run(a, child_context)
        event_wait_list = build_event_wait_list([a])
        dtype = tensor.data_type
        output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

        work_group = [a.total_elements]

        event = call_program(op_name, dtype, work_group, a.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        output_buffer.op = event
        output_buffer
      end

      def call_program(name, dtype, work_group, *args)
        _cl_program(name.to_s, dtype: dtype).send(:"#{name}_#{dtype}", _opencl_queue, work_group, *args)
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
        # puts "convert_to_opencl called for #{name}"
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
                                 OpenCLBuffer.allocate_narray_for_type(data_type, value[0].bytesize)
                               else
                                OpenCLBuffer.allocate_narray_for_type(data_type, narray_size)
                               end

                      return nil if buffer.nil?

                      cl_buffer = unless array_fast_empty?(value)
                                    cl_buffer_size = 1 if cl_buffer_size.zero?
                                    _opencl_context.create_buffer(cl_buffer_size * buffer.element_size)
                                  end

                      @context[:_cache][cache_key] = OpenCLBuffer.new(self, name: name, data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer)
                    end

        if data_type == :string
          value[0].each_byte.with_index do |c, index|
            cl_object.buffer[index] = c
          end
        elsif value.is_a?(Array)
          cast_value = value.flatten.each_with_index.map do |element, index|
           if element.is_a?(Tensor)
                                        read_final_result(complete_eval(element, {}))
                                      elsif data_type == :boolean
                                        element ? 1 : 0
                                      else
                                        Tensor.cast_dtype(element, data_type)
                                      end
          end

          cast_value.each_with_index do |v, index|
            cl_object.buffer[index] = v
          end
        elsif value.is_a?(NArray)
          cl_object.buffer = value
        elsif data_type == :boolean
          cl_object.buffer[0] = element ? 1 : 0
        else
          cl_object.buffer[0] = Tensor.cast_dtype(value, data_type)
        end

        # if OpenCL buffer is valid enqueue a write
        if cl_object.cl_buffer && value && (!value.is_a?(Array) || !value.empty?)
          cl_object.op = _opencl_queue.enqueue_write_buffer(cl_object.cl_buffer, cl_object.buffer)
        end

        cl_object
      end

      def _create_result_buffer(data_type, shape, name, allocate_host: false)
        return OpenCLBuffer.nil_buffer(self, name, data_type) if shape == [0]

        cache_key = "_result_#{name}_#{shape.join('_')}:#{object_id}"
        @context[:_cache][:_cl_buffers][cache_key] ||= begin
          # puts "create result buffer #{cache_key}"
          size = shape.empty? || shape == [0] ? 1 : shape.reduce(:*)
          lazy_buffer = !allocate_host ? OpenCLBuffer::LazyBuffer.new(data_type, size) : OpenCLBuffer.allocate_narray_for_type(data_type, size)
          cl_buffer = _opencl_context.create_buffer(size * lazy_buffer.element_size)

          OpenCLBuffer.new(self, data_type: data_type, shape: shape, buffer: lazy_buffer, cl_buffer: cl_buffer, name: name)
        end
      end

      # automatically use sub buffers
      def _create_result_sub_buffer(parent_buffer, index, data_type, shape, name)
        cache_key ="_sub_result_#{parent_buffer.object_id}_#{name}_#{index}:#{object_id}"
        @context[:_cache][:_cl_buffers][cache_key] ||= begin
          size = shape.empty? || shape == [0] ? 1 : shape.reduce(:*)
          buffer = OpenCLBuffer.allocate_narray_for_type(data_type, size)

          if parent_buffer.cl_buffer.associated_memobject.nil?
            start = index * buffer.size * buffer.element_size
            region = OpenCL::BufferRegion::new(start, buffer.size * buffer.element_size)
            cl_buffer = parent_buffer.cl_buffer.create_sub_buffer(OpenCL::BUFFER_CREATE_TYPE_REGION, region)
            OpenCLBuffer.new(self, data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer, name: name)
          else # source buffer already a sub buffer, OpenCL does not allow sub buffers from sub buffers
            _create_result_buffer(tensor.data_type, shape, name)
          end
        end

        buffer = @context[:_cache][:_cl_buffers][cache_key]

        if buffer.cl_buffer.associated_memobject
          buffer.op = parent_buffer.op
        else # source buffer alreay a sub buffer, so we need to do a copy instead
          region_size_in_bytes = buffer.buffer.size * buffer.buffer.element_size
          start = index * region_size_in_bytes
          region = [region_size_in_bytes, 1, 1]
          buffer.op = _opencl_queue.enqueue_copy_buffer_rect(parent_buffer.cl_buffer, buffer.cl_buffer, region, src_origin: [start, 0, 0], event_wait_list: parent_buffer.op)
        end

        buffer
      end

      # create sub buffers of different sizes
      def _create_variable_result_sub_buffer(parent_buffer, index, start, region_size_in_bytes, data_type, shape, name)
        cache_key = "_sub_result_#{parent_buffer.object_id}_#{name}_#{index}:#{object_id}"
        @context[:_cache][:_cl_buffers][cache_key] ||= begin
          size = shape.empty? || shape == [0] ? 1 : shape.reduce(:*)
          buffer = OpenCLBuffer.allocate_narray_for_type(data_type, size)

          if parent_buffer.cl_buffer.associated_memobject.nil?
            region = OpenCL::BufferRegion::new(start, region_size_in_bytes)
            cl_buffer = parent_buffer.cl_buffer.create_sub_buffer(OpenCL::BUFFER_CREATE_TYPE_REGION, region)
            OpenCLBuffer.new(self, data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer, name: "#{name}/sub")
          else
            _create_result_buffer(tensor.data_type, shape, name)
          end
        end

        buffer = @context[:_cache][:_cl_buffers][cache_key]

        if buffer.cl_buffer.associated_memobject
          buffer.op = parent_buffer.op
        else
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

      def resolve_placeholder(placeholder, _execution_context = {})
        return nil if placeholder.nil?
        return placeholder unless placeholder.is_a?(Placeholder)

        var = @context[placeholder.name.to_sym]
        raise "missing placeholder #{placeholder.name}" if var.nil?

        cache_key = "#{placeholder.graph.object_id}_opencl_#{placeholder.name}_p:#{object_id}"
        @context[cache_key] ||= begin
          convert_to_opencl(var, shape_eval(var), data_type: placeholder.data_type, name: placeholder.name) unless var.is_a?(Tensor)
        end
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

      ##
      # Fast way to determine if array is "empty" by including nested elements
      def array_fast_empty?(arr)
        return true if arr.size.zero?

        arr.each do |a|
          if a.is_a?(Array)
            return false if !array_fast_empty?(a)

            next
          end
          return false
        end

        true
      end
    end
  end
end

TensorStream::Evaluator.register_evaluator(TensorStream::Evaluator::OpenclEvaluator, 'opencl', 1)
