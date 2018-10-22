module TensorStream
  # Buffer used by the OpenCL evaluator
  class OpenCLBuffer < Buffer
    include ArrayOpsHelper

    attr_accessor :shape, :buffer, :cl_buffer, :op, :owner

    def initialize(owner, data_type:, shape:, buffer:, cl_buffer:, op: nil, name: nil)
      @data_type = data_type
      @shape = shape
      @buffer = buffer
      @cl_buffer = cl_buffer
      @name = name
      @op = op
      @owner = owner
    end

    def total_elements
      shape.reduce(:*) || 1
    end

    def empty_value?
      @shape == [0]
    end

    def to_ruby
      return [] if buffer.empty?

      if dirty
        op.command_queue.enqueue_read_buffer(cl_buffer, buffer, event_wait_list: [op].compact)
        op.command_queue.finish
        self.dirty = false
      end

      if shape.empty?
        return buffer.to_s if data_type == :string
        return buffer[0] != 0 if data_type == :boolean
        return buffer[0]
      end

      result = buffer.reshape(*shape.map(&:to_i).reverse).to_a
      data_type == :boolean ? process_function_op(result, ->(a, _b) { a != 0 }) : result
    end

    def self.nil_buffer(owner, name, data_type)
      OpenCLBuffer.new(owner, name: name, data_type: data_type, shape: [0], buffer: nil, cl_buffer: nil)
    end
  end
end
