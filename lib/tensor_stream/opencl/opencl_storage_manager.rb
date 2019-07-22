module TensorStream
  class OpenclStorageManager
    include TensorStream::CLEventHelpers

    def self.current_storage_manager
      @storage_manager ||= OpenclStorageManager.new
    end

    def initialize
      @variables = {}
    end

    def clear_variables(graph)
      @variables[graph.object_id] = {}
    end

    def exists?(graph, name)
      @variables.dig(graph.object_id, name.to_sym)
    end

    def read_value(graph, name)
      buffer = @variables.dig(graph.object_id, name.to_sym)

      raise "#{name} not initialized" unless buffer

      buffer[:buffer].to_ruby
    end

    def cl_read_var(graph, queue, name)
      device_id = device_id_key(queue)
      @variables[graph.object_id] ||= {}
      buffer = @variables.dig(graph.object_id, name.to_sym)
      raise "variable #{name} not yet initialized!" unless buffer

      buffer[:buffer]
    end

    def cl_assign_var(graph, queue, name, value_buffer)
      device_id = device_id_key(queue)
      @variables[graph.object_id] ||= {}
      buffer = @variables.dig(graph.object_id, name.to_sym)
      var_buffer = if buffer
        buffer[:buffer]
      else
        buffer = create_result_buffer(queue.context, value_buffer.data_type, value_buffer.shape, name)
        @variables[graph.object_id][name.to_sym] = {
          buffer: buffer,
          device: device_id
        }
        buffer
      end

      event_wait_list = build_event_wait_list([value_buffer, var_buffer])

      if value_buffer.cl_buffer != var_buffer
        var_buffer.dirty = true
        var_buffer.op = queue.enqueue_copy_buffer(value_buffer.cl_buffer, var_buffer.cl_buffer, event_wait_list: event_wait_list)
      end

      var_buffer
    end

    def create_result_buffer(opencl_context, data_type, shape, name, allocate_host: false)
      return OpenCLBuffer.nil_buffer(self, name, data_type) if shape == [0]

      # puts "create result buffer #{cache_key}"
      size = shape.empty? || shape == [0] ? 1 : shape.reduce(:*)
      lazy_buffer = !allocate_host ? OpenCLBuffer::LazyBuffer.new(data_type, size) : OpenCLBuffer.allocate_narray_for_type(data_type, size)
      cl_buffer = opencl_context.create_buffer(size * lazy_buffer.element_size)

      OpenCLBuffer.new(self, data_type: data_type, shape: shape, buffer: lazy_buffer, cl_buffer: cl_buffer, name: name)
    end

    protected

    def device_id_key(queue)
      "#{queue.device.name}_#{queue.device.pci_bus_id_nv}:#{queue.device.pci_slot_id_nv}"
    end
  end
end