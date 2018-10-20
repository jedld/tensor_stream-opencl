require "bundler/setup"
require 'tensor_stream'
require 'tensor_stream/opencl'
require 'pry-byebug'

ts = TensorStream

n = 10
DIMEN = 1024

A = ts.random_uniform([DIMEN, DIMEN]).eval
B = ts.random_uniform([DIMEN, DIMEN]).eval

# Create a graph to store results
c1 = []
c2 = []
a = nil
b = nil

def matpow(m, n)
  return m if n < 1
  TensorStream.matmul(m, matpow(m, n-1))
end

ts.device('/device:GPU:0') do
  a = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  b = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  # Compute A^n and B^n and store results in c1
  c1 << matpow(a, n)
  c1 << matpow(b, n)
end

sum = ts.device('/device:GPU:0') do
  ts.add_n(c1)
end

t1_1 = nil
t2_1 = nil
puts "===================== starting single GPU test ================"
ts.session(log_device_placement: true) do |sess|
  sess.run(sum, feed_dict: { a => A, b => B}) # warmup
  time = Time.now
  t1_1 = time.to_i * (10 ** 9) + time.nsec
  sess.run(sum, feed_dict: { a => A, b => B})
  time = Time.now
  t2_1 = time.to_i * (10 ** 9) + time.nsec
end

# Multi GPU computing
# GPU:0 computes A^n
ts.device('/device:GPU:0') do
  a = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  c2 << matpow(a, n)
end

# GPU:1 computes B^n
ts.device('/device:GPU:1') do
  b = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  c2 << matpow(b, n)
end

ts.device('/device:GPU:1') do
  sum = ts.add_n(c2) #Addition of all elements in c2, i.e. A^n + B^n
end

t1_2 = nil
t2_2 = nil

ts.session(log_device_placement: true) do |sess|
    # Run the op.
    sess.run(sum, feed_dict: {a => A, b => B}) # warm up
    time = Time.now
    t1_2 = time.to_i * (10 ** 9) + time.nsec
    puts "================ starting multiGPU test ==============="
    sess.run(sum, feed_dict: {a => A, b => B})
    time = Time.now
    t2_2 = time.to_i * (10 ** 9) + time.nsec
end


puts("Single GPU computation time: " + ((t2_1-t1_1)/ 1000000.to_f).to_s)
puts("Multi GPU computation time: " + ((t2_2-t1_2)/ 1000000.to_f).to_s)
