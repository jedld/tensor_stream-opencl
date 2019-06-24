module TensorStream
  class OpenclStorageManager
    def self.current_storage_manager
      @storage_manager ||= OpenclStorageManager.new
    end

    def initialize
      @variables = {}
    end

    def cl_assign_var(queue, name, cl_buffer)
      if @variables.key?(name.to_sym)
        event_wait_list = build_event_wait_list([cl_buffer, assign.container_buffer])
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