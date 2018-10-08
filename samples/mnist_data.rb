# A ruby port of the example code discussed by Martin Gorner in
# "TensorFlow and Deep Learning without a PhD, Part 1 (Google Cloud Next '17)""
#
# https://www.youtube.com/watch?v=u4alGiomYP4
#
# Requirements:
#   mnist-learn gem
#   opencl_ruby_ffi gem
require "bundler/setup"
require 'tensor_stream'
require 'mnist-learn'
# require 'pry-byebug'

# Enable OpenCL hardware accelerated computation, not using OpenCL can be very slow
require 'tensor_stream/opencl'

tf = TensorStream

# Import MNIST data
puts "downloading minst data"
mnist = Mnist.read_data_sets('/tmp/data', one_hot: true)
puts "downloading finished"

x = tf.placeholder(:float32, shape: [nil, 784])

K = 200
L = 100
M = 60
N = 30


w1 = tf.variable(tf.random_normal([784, K]))
b1 = tf.variable(tf.zeros([K]))

w2 = tf.variable(tf.random_normal([K, L]))
b2 = tf.variable(tf.zeros([L]))

w3 = tf.variable(tf.random_normal([L, M]))
b3 = tf.variable(tf.zeros([M]))

w4 = tf.variable(tf.random_normal([M, N]))
b4 = tf.variable(tf.zeros([N]))

w5 = tf.variable(tf.random_normal([N, 10]))
b5 = tf.variable(tf.zeros([10]))

x_ = tf.reshape(x, [-1, 784])

y1 = tf.sigmoid(tf.matmul(x_, w1) + b1)
y2 = tf.sigmoid(tf.matmul(y1, w2) + b2)
y3 = tf.sigmoid(tf.matmul(y2, w3) + b3)
y4 = tf.sigmoid(tf.matmul(y3, w4) + b4)

# model
y = tf.nn.softmax(tf.matmul(y4, w5) + b5)

y_ = tf.placeholder(:float32, shape: [nil, 10])

# loss function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy =  tf.reduce_mean(tf.cast(is_correct, :float32))

optimizer = TensorStream::Train::GradientDescentOptimizer.new(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.session
# Add ops to save and restore all the variables.
saver = tf::Train::Saver.new
init = tf.global_variables_initializer

sess.run(init)
mnist_train = mnist.train
(0...1000).each do |i|
  # load batch of images and correct answers
  batch_x, batch_y = mnist_train.next_batch(100)

  train_data = { x => batch_x, y_ => batch_y }

  # train
  sess.run(train_step, feed_dict: train_data)
  if (i % 10 == 0)
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in path: %s" % save_path)
    # success? add code to print it
    a, c = sess.run([accuracy, cross_entropy], feed_dict: train_data)
    puts "#{i} train accuracy #{a}, error #{c}"

    # success on test data?
    test_data = { x => mnist.test.images, y_ => mnist.test.labels }
    a, c = sess.run([accuracy, cross_entropy], feed_dict: test_data)
    puts " test accuracy #{a}, error #{c}"
  end
end

