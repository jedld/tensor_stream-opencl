module TensorStream
  # Buffer used by the OpenCL evaluator
  class OpenCLBuffer < Buffer
    class LazyBuffer
      attr_reader :data_type

      def initialize(data_type, size)
        @data_type = data_type
        @size = size
      end

      def size
        @size
      end

      def element_size
        buffer_size_for_type(@data_type)
      end

      def buffer_size_for_type(data_type)
        case data_type
        when :float, :float32, :float16
          4
        when :float64
          8
        when :int, :int32, :int64, :uint64, :uint32 # NArray does not have 64 bit int types
          4
        when :int16, :uint16
          2
        when :uint8, :int8
          1
        when :boolean
          1
        when :string
          1
        when :unknown
          nil
        else
          raise "unsupported type #{data_type}"
        end
      end
    end

    include ArrayOpsHelper
    include TensorStream::CLEventHelpers

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

    def inspect
      "CLBuffer(name: #{name} shape: #{shape || "?"} data_type: #{data_type}, cl_allocated: #{cl_buffer ? cl_buffer.size : 'unallocated'}) -> raw: #{buffer.to_a}"
    end

    def buffer!
      return buffer if buffer.is_a?(NArray)

      @buffer = OpenCLBuffer.allocate_narray_for_type(buffer.data_type, buffer.size) if buffer.is_a?(LazyBuffer)

      command_queue.enqueue_read_buffer(cl_buffer, @buffer, blocking: true, event_wait_list: build_event_wait_list([self]))
      @buffer
    end

    def command_queue
      @command_queue ||= begin
        first_op = op.is_a?(Array) ? op.first : op
        first_op.command_queue
      end
    end

    def to_ruby
      buffer! if buffer.is_a?(LazyBuffer)

      return [] if buffer.empty?

      if dirty
        command_queue.enqueue_read_buffer(cl_buffer, buffer, event_wait_list: [op].compact)
        command_queue.finish
        self.dirty = false
      end

      if shape.empty?
        return case data_type
               when :string
                 buffer.to_s
               when :boolean
                 buffer[0] != 0
               else
                 buffer[0]
               end
      end

      result = buffer.reshape(*shape.map(&:to_i).reverse).to_a
      data_type == :boolean ? process_function_op(result) { |a, _b|  a != 0 } : result
    end

    def self.nil_buffer(owner, name, data_type)
      OpenCLBuffer.new(owner, name: name, data_type: data_type, shape: [0], buffer: nil, cl_buffer: nil)
    end

    def self.allocate_narray_for_type(data_type, narray_size)
      case data_type
      when :float, :float32, :float16
        NArray.sfloat(narray_size)
      when :float64
        NArray.float(narray_size)
      when :int, :int32, :int64, :uint64, :uint32 # NArray does not have 64 bit int types
        NArray.int(narray_size)
      when :int16, :uint16
        NArray.sint(narray_size)
      when :uint8, :int8
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
  end
end
