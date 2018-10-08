[![Gem Version](https://badge.fury.io/rb/tensor_stream-opencl.svg)](https://badge.fury.io/rb/tensor_stream-opencl)

# TensorStream::Opencl

This gem provides an OpenCL backend for TensorStream (https://github.com/jedld/tensor_stream). OpenCL is an open standard
that allows running compute applications on heterogenous platforms like CPUs and GPUs. For certain neural network implementations, like deep neural networks GPU acceleration can dramatically speedup computation.

## Installation

Make sure OpenCL device drivers are installed in your system. You may refer to the following links:

### Nvidia

https://developer.nvidia.com/opencl

### AMD

https://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx


### Intel

https://software.intel.com/en-us/articles/opencl-drivers


Add this line to your application's Gemfile:

```ruby
gem 'tensor_stream-opencl'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install tensor_stream-opencl

## Usage

If using a Gemfile or a framework like rails, simply including this gem will allow tensor_stream to automatically select opencl devices for use in your computation. Otherwise you can do:

```ruby
require 'tensor_stream/opencl'
```

You can check for available OpenCL devices via'

```ruby
TensorStream::Evaluator::OpenclEvaluator.query_supported_devices

TensorStream::Evaluator::OpenclEvaluator.query_supported_devices.map(&:native_device)
# => [#<OpenCL::Device: Intel(R) Core(TM) i5-5575R CPU @ 2.80GHz (4294967295)>, #<OpenCL::Device: Intel(R) Iris(TM) Pro Graphics 6200 (16925952)>]
```

## Device placement control

You can place operations on certain devices using ts.device:

```ruby
require 'tensor_stream/opencl'

ts = TensorStream
# For the first GPU
ts.device('/device:GPU:0') do
  a = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  b = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  # Compute A^n and B^n and store results in c1
  c1 << matpow(a, n)
  c1 << matpow(b, n)
end

# For the second GPU
ts.device('/device:GPU:1') do
  a = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  b = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  # Compute A^n and B^n and store results in c1
  c1 << matpow(a, n)
  c1 << matpow(b, n)
end
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/jedld/tensor_stream-opencl. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the TensorStream::Opencl project’s codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/[USERNAME]/tensor_stream-opencl/blob/master/CODE_OF_CONDUCT.md).
