require "bundler/setup"
require "tensor_stream"
require "tensor_stream/opencl"
# require "tensor_stream/utils/tensor_utils"
require "pry-byebug"
require 'simplecov'
SimpleCov.start

Dir["../tensor_stream/spec/support/**/*.rb"].sort.each { |f| require f }

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end

# Helper function to truncate floating point values (for testing)
# truncation is done in tests since different machines return the last digits of
# fp values differently
def tr(t, places = 4)
  if t.is_a?(Array)
    return t.collect do |v|
      tr(v, places)
    end
  end

  return t unless t.kind_of?(Float)

  t.round(places)
end


def trf(t, places)
  if t.is_a?(Array)
    return t.collect do |v|
      trf(v, places)
    end
  end

  return t unless t.kind_of?(Float)
  t.truncate(places)
end