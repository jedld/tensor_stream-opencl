module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module RandomOps
      RAND_TABLE_SIZE = 1024 * 1024

      def RandomOps.included(klass)
        klass.class_eval do
          register_op :truncated_normal do |context, tensor, inputs|
            seed = tensor.options[:seed]
            random = _get_randomizer(tensor, seed)
            r = RandomGaussian.new(tensor.options.fetch(:mean), tensor.options.fetch(:stddev), -> { random.rand })
            generator = -> { r.rand }
            buffer = _randomizer(context, tensor, inputs, generator)
            buffer.sync!
            mean = buffer.buffer.mean
            random_values = buffer.buffer
            stddev = Math.sqrt(random_values.map { |v| (v - mean)**2 }.sum / (random_values.size - 1))
            minval = random_values.min
            maxval = random_values.max
            max_iterations = 100

            if (minval.infinite? && minval < 0.0) || (maxval < mean)
              # Reverse all calculations. normMin and normMax will be flipped.
              a = minval
              minval = maxval
              maxval = a
              stddev = -stddev
            end

            norm_min = (minval - mean) / stddev
            norm_max = (maxval - mean) / stddev

            val = random_values.map { |v|
              iterations = 0
              pick = v
              while (pick > norm_max) || (pick < norm_min)
                pick = generator.call
                iterations += 1
                if iterations > max_iterations
                  pick = v
                  break
                end
              end

              pick
            }
            buffer.buffer = val
            buffer.op = _opencl_queue.enqueue_write_buffer(buffer.cl_buffer, buffer.buffer)
            buffer
          end

          register_op :random_uniform do |context, tensor, inputs|
            seed = tensor.options[:seed]
            random = _get_randomizer(tensor, seed)
            generator = ->() { random.rand }
            _randomizer(context, tensor, inputs, generator)
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

          def _randomizer(context, tensor, inputs, func)
            maxval = tensor.options.fetch(:maxval, 1)
            minval = tensor.options.fetch(:minval, 0)
            seed = tensor.options[:seed]
            table_size = tensor.options[:pre_gen_table_size] || RAND_TABLE_SIZE

            seed_ptr_key_name = "_rand_seed_ptr_#{tensor.operation}_#{seed}"
            rand_buffer = @context[:_cache][:_cl_buffers]["_rand_#{tensor.operation}_#{seed}"] ||= begin
              @context[:_cache][:_cl_buffers][seed_ptr_key_name] = 0
              rand_table = table_size.times.map { func.call }
              convert_to_opencl(rand_table, [table_size], data_type: tensor.data_type, name: "rand_#{tensor.operation}_#{seed}_#{tensor.data_type}")
            end

            @context[:_cache][:_cl_buffers][seed_ptr_key_name] ||= 0

            seed_ptr = @context[:_cache][:_cl_buffers][seed_ptr_key_name]

            shape = read_final_result(complete_eval(inputs[0], context))
            shape = shape || tensor.shape.shape
            workgroup = [shape.reduce(:*) || 1 ]
            cl_seed_ptr = OpenCL::Int1.new(seed_ptr)
            cl_min = OpenCL::Float1.new(minval)
            cl_max = OpenCL::Float1.new(maxval)

            @context[:_cache][:_cl_buffers][seed_ptr_key_name] = (seed_ptr + (shape.reduce(:*) || 0) ) % RAND_TABLE_SIZE
            buffer = _create_result_buffer(tensor.data_type, shape, tensor.name)
            buffer.op = _cl_program("random_uniform", dtype: tensor.data_type, tsize: RAND_TABLE_SIZE).send(:"random_uniform_#{tensor.data_type}", _opencl_queue, workgroup, cl_seed_ptr, cl_min, cl_max, rand_buffer.cl_buffer, buffer.cl_buffer)
            buffer
          end
        end
      end
    end
  end
end