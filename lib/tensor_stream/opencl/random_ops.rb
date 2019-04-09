module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module RandomOps
      RAND_TABLE_SIZE = 1024 * 1024

      def RandomOps.included(klass)
        klass.class_eval do
          register_op :random_uniform do |context, tensor, inputs|
            maxval = tensor.options.fetch(:maxval, 1)
            minval = tensor.options.fetch(:minval, 0)
            seed = tensor.options[:seed]

            rand_buffer = @context[:_cache][:_cl_buffers]["_rand"] ||= begin
              @context[:_cache][:_cl_buffers]["_rand_seed_ptr"] = 0
              random = _get_randomizer(tensor, seed)
              rand_table = RAND_TABLE_SIZE.times.map { random.rand }
              convert_to_opencl(rand_table, [RAND_TABLE_SIZE], data_type: tensor.data_type, name: "rand_#{tensor.data_type}")
            end
            @context[:_cache][:_cl_buffers]["_rand_seed_ptr"] ||= 0

            seed_ptr = @context[:_cache][:_cl_buffers]["_rand_seed_ptr"]

            shape = read_final_result(complete_eval(inputs[0], context))
            shape = shape || tensor.shape.shape
            workgroup = [shape.reduce(:*) || 1 ] 
            cl_seed_ptr = OpenCL::Int1.new(seed_ptr)
            cl_min = OpenCL::Float1.new(minval)
            cl_max = OpenCL::Float1.new(maxval)

            @context[:_cache][:_cl_buffers]["_rand_seed_ptr"] = (seed_ptr + (shape.reduce(:*) || 0) ) % RAND_TABLE_SIZE
            buffer = _create_result_buffer(tensor.data_type, shape, tensor.name)
            buffer.op = _cl_program("random_uniform", dtype: tensor.data_type, tsize: RAND_TABLE_SIZE).send(:"random_uniform_#{tensor.data_type}", _opencl_queue, workgroup, cl_seed_ptr, cl_min, cl_max, rand_buffer.cl_buffer, buffer.cl_buffer)
            buffer
          end

          def _get_randomizer(tensor, seed)
            if tensor.graph.random_seed && seed
              Random.new(tensor.graph.random_seed ^ seed)
            elsif tensor.graph.random_seed
              @session.randomizer[tensor.graph.object_id] ||= Random.new(tensor.graph.random_seed)
              @session.randomizer[tensor.graph.object_id]
            elsif seed
              @session.randomizer[tensor.operation] ||= Random.new(seed)
              @session.randomizer[tensor.operation]
            else
              Random.new
            end
          end
        end
      end
    end
  end
end