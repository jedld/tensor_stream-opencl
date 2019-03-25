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

puts TensorStream::Evaluator.default_evaluators

sess2 = tf.session

if os == :macosx
  puts `sysctl -n machdep.cpu.brand_string`
else
  puts `cat /proc/cpuinfo | grep "model name" | head -1`
end
device = TensorStream::Evaluator::OpenclEvaluator.default_device.native_device
puts "OpenCL device #{device.platform.to_s} #{device.name}"
Benchmark.bmbm do |x|
  x.report("ruby argmin            :") { 100.times do sess.run(argmin) end }
  x.report("opencl argmin          :") { 100.times do sess2.run(argmin) end }
  x.report("ruby bias_add_grad     :") { 100.times do sess.run(bias_add_grad) end }
  x.report("opencl bias_add_grad   :") { 100.times do sess2.run(bias_add_grad) end }
  x.report("ruby bias_add          :") { 100.times do sess.run(bias_add) end }
  x.report("opencl bias_add        :") { 100.times do sess2.run(bias_add) end }
  x.report("ruby conv2d_backprop   :") { 100.times do sess.run(conv2d_grad) end }
  x.report("opencl conv2d_backprop :") { 100.times do sess2.run(conv2d_grad) end }
  x.report("ruby conv2d            :") { 100.times do sess.run(conv2d) end }
  x.report("opencl conv2d          :") { 100.times do sess2.run(conv2d) end }
  x.report("ruby arr index         :") { 100.times do sess.run(index) end }
  x.report("opencl arr index       :") { 100.times do sess2.run(index) end }
  x.report("ruby min               :") { 100.times do sess.run(min) end }
  x.report("opencl min             :") { 100.times do sess2.run(min) end }
  x.report("ruby sum               :") { 100.times do sess.run(sum) end }
  x.report("opencl sum             :") { 100.times do sess2.run(sum) end }
  x.report("ruby sum axis 1        :") { 100.times do sess.run(sum_axis_1) end }
  x.report("opencl sum axis 1      :") { 100.times do sess2.run(sum_axis_1) end }
  x.report("ruby split             :") { 100.times do sess.run(split) end }
  x.report("opencl split           :") { 100.times do sess2.run(split) end }
  x.report("ruby add_n             :") { 100.times do sess.run(add_n) end }
  x.report("opencl add_n           :") { 100.times do sess2.run(add_n) end }
  x.report("ruby ooo matmul        :") { 100.times do sess.run(out_of_order) end }
  x.report("opencl    ooo matmul   :") { 100.times do sess2.run(out_of_order) end }
  x.report("ruby softmax           :") { 100.times do sess.run(softmax) end }
  x.report("opencl    softmax      :") { 100.times do sess2.run(softmax) end }
  x.report("ruby matmul            :") { 100.times do sess.run(matmul) end }
  x.report("opencl    matmul       :") { 100.times do sess2.run(matmul) end }
  x.report("ruby                   :") { 100.times do sess.run(model, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl                 :") { 100.times do sess2.run(model, feed_dict: { p => rand, q => rand }) end }
  x.report("ruby single function   :") { 100.times do sess.run(single_function_test, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl single function :") { 100.times do sess2.run(single_function_test, feed_dict: { p => rand, q => rand }) end }
  x.report("ruby pow float         :") { 100.times do sess.run(pow_f, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl pow float       :") { 100.times do sess2.run(pow_f, feed_dict: { p => rand, q => rand }) end }
  x.report("ruby pow int           :") { 100.times do sess.run(pow_i, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl pow int         :") { 100.times do sess2.run(pow_i, feed_dict: { p => rand, q => rand }) end }
end