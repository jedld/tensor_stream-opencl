module TensorStream
  class OpenCLUtil
    ##
    # initializes a OpenCL helper class based on a session
    def initialize(session)
      @session = session
    end

    ##
    # Retrieves OpenCL memory usage
    def get_memory_usage
      cl_buffer_uniq_set = Set.new
      @session.last_session_context[:_cache][:_cl_buffers].inject(0) do |sum, elem|
        cl_buffer_uniq_set.add?(elem[1].cl_buffer.object_id) ? sum + elem[1].cl_buffer.size : sum
      end
    end
  end
end