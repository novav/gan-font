import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


width = 224
height = 224
pic_size = 224*224

mb_size = 160
Z_dim = 100

def plot(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(width, height), cmap='Greys_r')
    return fig


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=options)
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([1], tf.string)
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image

image = read_and_decode('./tfrecords/train.tfrecord')

# image_batch = tf.train.shuffle_batch([image], batch_size=mb_size, capacity = 50000, min_after_dequeue=10000, num_threads=1)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    tf.train.start_queue_runners()
    # 创建一个协调器，管理线程
    # coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0

    try:
        while True:
            b_image = sess.run([image])
            i += 1
    except tf.errors.OutOfRangeError:
        print('numeres = ' + i)
    finally:
        print('item i error', i)

    # b_image = np.reshape(b_image, (-1, width * height))

    print('===============')

exit()
