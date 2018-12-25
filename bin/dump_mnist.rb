#!/usr/bin/env ruby

require "bundler/setup"
require "tensor_stream"
require 'mnist-learn'
require 'pry-byebug'
require 'fileutils'

mnist = Mnist.read_data_sets('/tmp/data', one_hot: true)

ts = TensorStream
test_data = mnist.test.images
FileUtils.mkdir_p 'test_images'

sess = ts.session

test_data.each_with_index do |image , index|
  image = 255 - ts.cast(ts.reshape(image, [28, 28, 1]), :uint8) # reshape image
  encoder = ts.image.encode_png(image)
  blob = sess.run(encoder)
  File.write(File.join('test_images', "#{index}_image.png"), blob)
end