#!/usr/bin/env ruby

# builds a frozen mnist image recognition model from scratch
# based on mnist_data_3.0.rb

require "bundler/setup"
require 'tensor_stream'
require 'mnist-learn'
require 'csv'

# Enable OpenCL hardware accelerated computation, not using OpenCL can be very slow
require 'tensor_stream/opencl'
require 'tensor_stream/utils/freezer'

epochs = (ARGV[0] || 2000).to_i

tf = TensorStream
puts "Tensorstream version #{tf.__version__} with OpenCL lib #{TensorStream::Opencl::VERSION}"

# Import MNIST data
puts "downloading minst data"
# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = Mnist.read_data_sets('/tmp/data', one_hot: true)

puts "downloading finished"

x = tf.placeholder(:float32, shape: [nil, 28, 28, 1])
y_ = tf.placeholder(:float32, shape: [nil, 10])
step = tf.placeholder(:int32)
pkeep = tf.placeholder(tf.float32)

# Network parameters
K = 4 # first convolutional layer output depth
L = 8 # second convolutional layer output depth
M = 12 # third convolutional layer
N = 200 # fully connected layer


w1 = tf.variable(tf.truncated_normal([6, 6, 1, K], stddev: 0.1))
b1 = tf.variable(tf.ones([K])/10)

w2 = tf.variable(tf.truncated_normal([5, 5, K, L], stddev: 0.1))
b2 = tf.variable(tf.ones([L])/10)

w3 = tf.variable(tf.truncated_normal([4, 4, L, M], stddev: 0.1))
b3 = tf.variable(tf.ones([M])/10)

w4 = tf.variable(tf.truncated_normal([7 * 7 * M, N], stddev: 0.1))
b4 = tf.variable(tf.ones([N])/10)

w5 = tf.variable(tf.truncated_normal([N, 10], stddev: 0.1))
b5 = tf.variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
y1 = tf.nn.relu(tf.nn.conv2d(tf.reshape(x, [-1, 28, 28, 1]), w1, [1, stride, stride, 1], 'SAME') + b1)
stride = 2  # output is 14x14
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, [1, stride, stride, 1], 'SAME') + b2)
stride = 2  # output is 7x7
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, [1, stride, stride, 1], 'SAME') + b3)

# reshape the output from the third convolution for the fully connected layer
yy = tf.reshape(y3, [-1, 7 * 7 * M])
y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)

ylogits = tf.matmul(y4, w5) + b5

# model
y = tf.nn.softmax(ylogits, name: 'out')

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

init = tf.global_variables_initializer

sess.run(init)

#Setup save and restore
model_save_path = "test_models/mnist_data_3.0"
saver = tf::Train::Saver.new
saver.restore(sess, model_save_path)

mnist_train = mnist.train
test_data = { x => mnist.test.images, y_ => mnist.test.labels, pkeep => 1.0 }

puts "training #{epochs} epochs"
(0..epochs).each do |i|
  # load batch of images and correct answers
  batch_x, batch_y = mnist_train.next_batch(100)
  train_data = { x => batch_x, y_ => batch_y, step => i, pkeep => 0.75 }

  # train
  sess.run(train_step, feed_dict: train_data)

  if (i % 10 == 0)

    a_train, c_train, l = sess.run([accuracy, cross_entropy, lr], feed_dict: { x => batch_x, y_ => batch_y, step => i, pkeep => 1.0})
    puts "#{i}: accuracy:#{a_train} loss:#{c_train} (lr:#{l})"
  end

  if (i % 100 == 0)
    # success on test data?
    a_test, c_test = sess.run([accuracy, cross_entropy], feed_dict: test_data, pkeep => 1.0)
    puts("#{i}: ******** test accuracy: #{a_test} test loss: #{c_test}")

    # save current state of the model
    save_path = saver.save(sess, model_save_path)
  end
end

# Freezing removes variables and turns them into constants
puts "saving frozen model"
freezer = TensorStream::Freezer.new
freezer.convert(sess, model_save_path, 'mnist_model.yml')
puts "done. model written at mnist_model.yml"


