#!/usr/bin/env ruby

require "bundler/setup"
require "tensor_stream"
require 'mnist-learn'
require 'fileutils'

file_path = ARGV[0]
model_path = ARGV[1]

decoded_image = TensorStream.image.decode_png(File.read(file_path), channels: 1)
target_graph = TensorStream::YamlLoader.new.load_from_file(model_path)
input = target_graph['Placeholder']
output = TensorStream.argmax(target_graph['out'], 1)
sess = TensorStream.session

reshaped_image = 255.0.t - decoded_image.reshape([1, 28, 28, 1]).cast(:float32)
result = sess.run(output, feed_dict: { input => reshaped_image})

puts "image is a #{result.first}"

