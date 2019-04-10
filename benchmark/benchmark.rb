require "bundler/setup"
require 'tensor_stream'
require 'benchmark'
require 'pry-byebug'
require 'awesome_print'
require 'tensor_stream/opencl'
require 'rbconfig'

def os
  @os ||= (
    host_os = RbConfig::CONFIG['host_os']
    case host_os
    when /mswin|msys|mingw|cygwin|bccwin|wince|emc/
      :windows
    when /darwin|mac os/
      :macosx
    when /linux/
      :linux
    when /solaris|bsd/
      :unix
    else
      raise Error::WebDriverError, "unknown os: #{host_os.inspect}"
    end
  )
end

def tr(t, places = 1)
  if t.is_a?(Array)
    return t.collect do |v|
      tr(v, places)
    end
  end

  return t unless t.is_a?(Float)

  t.round(places)
end

tf = TensorStream

srand(5)
seed = 5
tf.set_random_seed(seed)

SHAPES = [32, 32]

sess = tf.session(:ruby_evaluator)
large_tensor = tf.constant(sess.run(tf.random_uniform([256, 256])))
a = tf.constant(sess.run(tf.random_uniform(SHAPES)))
a_int = tf.constant([
  [1, 2, 3, 4, 4, 1, 4, 8, 3, 4, 1, 1],
  [2, 2, 3, 4, 4, 1, 1, 1, 1, 4, 1, 1],
  [3, 2, 3, 4, 0, 1, 1, 2, 1, 1, 2, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 1, 1, 3, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 1, 1, 4, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 1, 5, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 1, 6, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 0, 0, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 6, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 1, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 1, 2],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 1, 2],
])

large_tensor_bias = tf.constant(sess.run(tf.random_uniform([256])))

b = tf.constant(sess.run(tf.random_uniform(SHAPES)))

c = tf.constant(sess.run(tf.random_uniform(SHAPES)))

d = tf.constant(sess.run(tf.random_uniform(SHAPES)))

sample_image = tf.constant(sess.run(tf.random_uniform([10, 8, 8, 3])))
sample_filter = tf.constant(sess.run(tf.random_uniform([2, 2, 3, 3])))

p = tf.placeholder('float')
q = tf.placeholder('float')

model = -tf.sin(a.dot(b + p) + c).dot(a) + tf.cos(a.dot(d + q))
single_function_test = (tf.sigmoid(a * p) * tf.sigmoid(b * q)) + c
pow_f = tf.pow(a, 3)
pow_i = tf.pow(a_int, 3)
matmul = tf.matmul(a, b)
out_of_order = tf.matmul(a, b) + tf.matmul(c, d)
softmax = tf.nn.softmax(a)
add_n = tf.add_n([a,b,c,d])
split = tf.split(a, 4)
sum = tf.reduce_sum(large_tensor)
sum_axis_1 = tf.reduce_sum(large_tensor, 1)
min = tf.min(large_tensor, 1)
argmin = tf.argmin(large_tensor)
index = large_tensor[0]

conv2d = tf.nn.conv2d(sample_image, sample_filter, [1, 1, 1, 1], 'SAME')
conv2d_grad = tf.gradients(conv2d, [sample_image, sample_filter])

bias_add = tf.nn.bias_add(large_tensor, large_tensor_bias)
bias_add_grad = tf.gradients(bias_add, [large_tensor_bias])
dropout = tf.nn.dropout(large_tensor, 0.8)

puts TensorStream::Evaluator.default_evaluators

sess2 = tf.session

cpu = if os == :macosx
          `sysctl -n machdep.cpu.brand_string`
      else
        `cat /proc/cpuinfo | grep "model name" | head -1`
      end

device = TensorStream::Evaluator::OpenclEvaluator.default_device.native_device
cl_device =  "#{device.platform.to_s} #{device.name}"

tests = {
  "argmin" => argmin,
  "bias_add_grad" => bias_add_grad,
  "bias_add" => bias_add,
  "conv2d_backprop" => conv2d_grad,
  "conv2d" => conv2d,
  "index" =>index,
  "min" => min,
  "sum" => sum,
  "sum axis 1" => sum_axis_1,
  "split" => split,
  "add_n" => add_n,
  "out of order matmul" => out_of_order,
  "softmax" => softmax,
  "matmul" => matmul,
  "test model" => ->(sess) { sess.run(model, feed_dict: { p => rand, q => rand }) },
  "single function test" => ->(sess) { sess.run(single_function_test, feed_dict: { p => rand, q => rand }) },
  "pow (float)" => ->(sess) { sess.run(pow_f, feed_dict: { p => rand, q => rand }) },
  "pow (int)" => ->(sess) { sess.run(pow_i, feed_dict: { p => rand, q => rand }) },
  "dropout" => dropout
}

stats = {
  ruby: {},
  opencl: {},
}

puts "rehersal"
tests.each do |k, v|
  if v.is_a?(Proc)
      r = Benchmark.measure("ruby #{k}") { 10.times do v.call(sess) end }
      r =Benchmark.measure("opencl #{k}") { 10.times do v.call(sess2) end }
  else
      r = Benchmark.measure("ruby #{k}") { 10.times do sess.run(v) end }
      r = Benchmark.measure("opencl #{k}") { 10.times do sess2.run(v)  end }
  end
end

puts "writing benchmark"

tests.each do |k, v|
  if v.is_a?(Proc)
    r = Benchmark.measure(k) { 100.times do v.call(sess) end }
    stats[:ruby][r.label] = { real: r.real, stime: r.stime, total: r.total, utime: r.utime }
    r = Benchmark.measure(k) { 100.times do v.call(sess2) end }
    stats[:opencl][r.label] = { real: r.real, stime: r.stime, total: r.total, utime: r.utime }
  else
    r = Benchmark.measure(k) { 100.times do sess.run(v) end }
    stats[:ruby][r.label] = { real: r.real, stime: r.stime, total: r.total, utime: r.utime }
    r = Benchmark.measure(k) { 100.times do sess2.run(v)  end }
    stats[:opencl][r.label] = { real: r.real, stime: r.stime, total: r.total, utime: r.utime }
  end
end

output = {
  "#{cpu.strip.gsub("model name\t: ", "")} #{cl_device.strip}" => stats
}

File.write("benchmark_#{Time.now.strftime('%Y%m%d%H%M')}.json", output.to_json)
