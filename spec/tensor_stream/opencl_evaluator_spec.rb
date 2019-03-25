require "spec_helper"
require 'benchmark'
require 'tensor_stream/opencl/opencl_evaluator'
require 'tensor_stream'

RSpec.describe TensorStream::Evaluator::OpenclEvaluator do
  let(:tf) { TensorStream }
  let(:sess) { TensorStream.session([:opencl_evaluator, :ruby_evaluator]) }
  let(:instance) { described_class.new(sess, TensorStream::Evaluator::OpenclEvaluator.default_device, {})}

  it_behaves_like "standard ops evaluator"
  it_behaves_like "optimizer evaluator"
  it_behaves_like "images ops"
  it_behaves_like "standard nn ops evaluator"
  it_behaves_like "TensorStream::Train::Saver"

  describe "supported TensorStream version" do
    it "returns the version" do
      expect(TensorStream.version).to eq("1.0.7")
    end
  end

  def create_session
    TensorStream.session([:opencl_evaluator, :ruby_evaluator])
  end

  context ".query_device" do
    it "selects the first GPU if there is one" do
      device = described_class.query_device("/device:GPU:0")
      expect(device).to be
      expect(device.type).to eq(:gpu)
      device = described_class.query_device("/gpu:0")
      expect(device).to be
      expect(device.type).to eq(:gpu)
    end

    context "tensor_stream convention" do
      it "selects a specific device on evaluator" do
      devices = tf.list_local_devices.select { |d| d =~ /opencl/ }
      device = described_class.query_device(devices.first)
      expect(device).to be
      end
    end
  end

  context "device placement test" do
    it "should evaluate tensors in appropriate device" do
      sess = TensorStream.session([:opencl_evaluator, :ruby_evaluator], log_device_placement: true)
      c = tf.device("/cpu:0") do
        a = tf.constant(1.0)
        b = tf.constant(1.0)
        a + b
      end
      expect(c.device).to eq("/cpu:0")
      d = tf.device("/device:GPU:0") do
        a = tf.constant(1.0)
        b = tf.constant(1.0)
        a + b
      end
      expect(d.device).to eq("/device:GPU:0")
      sess.run(c, d)
    end
  end

  context "supported ops" do
    specify do
      expect(described_class.ops.keys.size).to eq(108)
    end

    specify do
      expect(described_class.ops.keys.sort).to eq(%i[
        abs
        acos
        add
        add_n
        apply_adadelta
        apply_adagrad
        apply_adam
        apply_centered_rms_prop
        apply_gradient_descent
        apply_momentum
        apply_rms_prop
        argmax
        argmin
        asin
        assign
        assign_add
        assign_sub
        bias_add
        bias_add_grad
        broadcast_gradient_args
        broadcast_transform
        case
        cast
        ceil
        check_numerics
        concat
        cond
        const
        conv2d
        conv2d_backprop_filter
        conv2d_backprop_input
        cos
        decode_png
        div
        encode_png
        equal
        exp
        expand_dims
        fill
        floor
        floor_div
        floor_mod
        flow_group
        greater
        greater_equal
        identity
        index
        less
        less_equal
        log
        log1p
        log_softmax
        logical_and
        mat_mul
        max
        mean
        min
        mod
        mul
        negate
        no_op
        not_equal
        ones
        ones_like
        placeholder
        pow
        print
        prod
        range
        rank
        real_div
        reciprocal
        relu6
        reshape
        restore_ts
        round
        shape
        shape_n
        sigmoid
        sigmoid_grad
        sign
        sin
        size
        slice
        softmax
        softmax_cross_entropy_with_logits_v2
        softmax_cross_entropy_with_logits_v2_grad
        softmax_grad
        sparse_softmax_cross_entropy_with_logits
        split
        sqrt
        square
        squared_difference
        squeeze
        stack
        stop_gradient
        sub
        sum
        tan
        tanh
        tanh_grad
        transpose
        unstack
        variable
        variable_v2
        where
        zeros
        zeros_like
      ])
    end

    it "allows automatic fallback" do
      a = tf.constant([1,2,3,4], dtype: :float32)
      c = tf.concat([a], 0)
      d = tf.sin(c)
      expect(tr(sess.run(d))).to eq([0.8415, 0.9093, 0.1411, -0.7568])
    end
  end

  context ".list_local_devices" do
    specify do
      expect(tf.list_local_devices.size > 1).to be
    end
  end

  describe "data types" do
    TensorStream::Ops::INTEGER_TYPES.each do |dtype|
      context "#{dtype}" do
        specify do
          a = tf.constant([1, 2, 3, 4, 5], dtype: dtype)
          b = tf.constant([5, 6, 7, 8, 9], dtype: dtype)
          f = a + b
          g = a * b
          h = a / b
          j = a % b

          expect(sess.run(f, g, h, j)).to eq([[6, 8, 10, 12, 14], [5, 12, 21, 32, 45], [0, 0, 0, 0, 0], [1, 2, 3, 4, 5]])
        end
      end
    end

    TensorStream::Ops::FLOATING_POINT_TYPES.each do |dtype|
      context "#{dtype}" do
        specify do
          a = tf.constant([1, 2, 3, 4, 5], dtype: dtype)
          b = tf.constant([5, 6, 7, 8, 9], dtype: dtype)
          f = a + b
          g = a * b
          h = a / b
          j = a - b
          expect(f.dtype).to eq(dtype)
          expect(tr(sess.run(f, g, h, j))).to eq([[6.0, 8.0, 10.0, 12.0, 14.0],
            [5.0, 12.0, 21.0, 32.0, 45.0],
            [0.2, 0.3333, 0.4286, 0.5, 0.5556],
            [-4.0, -4.0, -4.0, -4.0, -4.0]])
        end
      end
    end
  end
end