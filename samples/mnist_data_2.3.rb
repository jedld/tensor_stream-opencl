# A ruby port of the example code discussed by Martin Gorner in
# "TensorFlow and Deep Learning without a PhD, Part 1 (Google Cloud Next '17)""
#
# Five Layers with relu decay
# https://www.youtube.com/watch?v=u4alGiomYP4
#
# Requirements:
#   mnist-learn gem
#   opencl_ruby_ffi gem
require "bundler/setup"
require 'tensor_stream'
require 'mnist-learn'
require 'pry-byebug'

# Enable OpenCL hardware accelerated computation, not using OpenCL can be very slow
require 'tensor_stream/opencl'

tf = TensorStream

# Import MNIST data
puts "downloading minst data"
mnist = Mnist.read_data_sets('/tmp/data', one_hot: true)
puts "downloading finished"

x = tf.placeholder(:float32, shape: [nil, 784])
y_ = tf.placeholder(:float32, shape: [nil, 10])

# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(:int32)

K = 200
L = 100
M = 60
N = 30


w1 = tf.variable(tf.truncated_normal([784, K], stddev: 0.1))
b1 = tf.variable(tf.ones([K])/10)

w2 = tf.variable(tf.truncated_normal([K, L], stddev: 0.1))
b2 = tf.variable(tf.ones([L])/10)

w3 = tf.variable(tf.truncated_normal([L, M], stddev: 0.1))
b3 = tf.variable(tf.ones([M])/10)

w4 = tf.variable(tf.truncated_normal([M, N], stddev: 0.1))
b4 = tf.variable(tf.ones([N])/10)

w5 = tf.variable(tf.truncated_normal([N, 10], stddev: 0.1))
b5 = tf.variable(tf.zeros([10]))

x_ = tf.reshape(x, [-1, 784])

y1 = tf.nn.relu(tf.matmul(x_, w1) + b1)
y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)
ylogits = tf.matmul(y4, w5) + b5

# model
y = tf.nn.softmax(ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits: ylogits, labels: y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy =  tf.reduce_mean(tf.cast(is_correct, :float32))

# training step, learning rate = 0.003
lr = 0.0001.t +  tf.train.exponential_decay(0.003, step, 2000, 1/Math::E)
train_step = TensorStream::Train::AdamOptimizer.new(lr).minimize(cross_entropy)

sess = tf.session
# Add ops to save and restore all the variables.
saver = tf::Train::Saver.new
init = tf.global_variables_initializer

sess.run(init)
mnist_train = mnist.train
test_data = { x => mnist.test.images, y_ => mnist.test.labels, pkeep => 1.0 }

(0..1000).each do |i|
  # load batch of images and correct answers
  batch_x, batch_y = mnist_train.next_batch(100)
  train_data = { x => batch_x, y_ => batch_y, step => i, pkeep => 0.75 }

  # train
  sess.run(train_step, feed_dict: train_data)

  if (i % 50 == 0)
    # success? add code to print it
    a_train, c_train = sess.run([accuracy, cross_entropy], feed_dict: train_data)

    # success on test data?
    a_test, c_test = sess.run([accuracy, cross_entropy], feed_dict: test_data)
    puts "#{i} train accuracy #{a_train}, error #{c_train} test accuracy #{a_test}, error #{c_test}"
  end
end

