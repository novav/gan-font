import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


width = 224
height = 224
pic_size = 224*224

mb_size = 20
Z_dim = 100


def variable_init(size):
    in_dims = size[0]
    w_stddev = 1./ tf.sqrt(in_dims/2.)
    return tf.random_normal(shape=size, stddev = w_stddev)

# def net_build():

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def linear(input_, output_dim, name, stddev=0.02):
    with tf.name_scope(name):
        print('liner input_', input_.shape[-1])
        print('liner input_2', input_.shape.as_list()[-1])
        matrix = tf.Variable(tf.random_normal(shape=[input_.shape.as_list()[-1], output_dim], stddev=stddev, dtype=tf.float32))
        baise = tf.Variable(tf.zeros([output_dim]))
        return tf.matmul(input_, matrix) + baise

def batch_norm(x, name, train=True, epsilon=1e-5, momentum=0.9):
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                   scale=True, is_training=train, scope=name)

img_height = height
img_width = width

out_height = height
out_width = width
batch_size = mb_size
c_dim = 128
y_dim = 10
df_dim = 64
dfc_dim = 1024
gf_dim = 64
gfc_dim = 1024
max_epoch = 300
z_dim = 100


def deconvolution(input_, output_dim, name, k_h = 5, k_w=5, s_h=2, s_w =2, stddev=0.02):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=[k_h, k_w, output_dim[-1], input_.shape.as_list()[-1]], stddev=stddev, dtype=tf.float32))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_dim, strides=[1, s_h, s_w, 1])
        b= tf.Variable(tf.zeros([output_dim[-1]]))
        return tf.reshape(tf.nn.bias_add(deconv, b), deconv.shape)


def generator(z):
    """

    :param z:   (100)
    :return:    (784)
    """

    s_h, s_w = out_height, out_width
    s_h2, s_w2 = int(s_h / 2), int(s_w / 2)
    s_h4, s_w4 = int(s_h / 4), int(s_w / 4)

    # h0 = tf.nn.relu(batch_norm(linear(z, gfc_dim, name='g_fc'), name='g_fcb1'))
    print('--z--', z)
    ll = linear(z, gfc_dim, name='g_fc')
    print('----', ll)
    bn = batch_norm(ll, name='g_fc_bn')
    print('----bb ', bn)

    h0 = tf.nn.relu(bn)

    h1 = tf.nn.relu(batch_norm(linear(h0, gf_dim * 2 * s_h4 * s_w4, name='g_fc2'), name='g_fc2_bn'))
    h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])

    print('gen_h1 ', h1)
    h2 = tf.nn.relu(batch_norm(deconvolution(h1, [batch_size, s_h4, s_w4, gf_dim * 2], name='g_dc', s_h=1, s_w=1), name='g_dc_bn'))
    print('gen_h2 ', h2)
    h_1 = tf.nn.relu(batch_norm(deconvolution(h2, [batch_size, s_h4, s_w4, gf_dim * 2], name='g_dc_1', s_h=1, s_w=1), name='g_dc_bn_1'))
    print('gen_h_1', h_1)
    h_2 = tf.nn.relu(batch_norm(deconvolution(h_1, [batch_size, s_h2, s_w2, gf_dim * 2], name='g_dc_2', ), name='g_dc_bn_2'))
    print('gen_h_2', h_2)

    # return tf.nn.sigmoid(deconvolution(h_2, [batch_size,  s_h2,    s_w,    c_dim],name='g_dc2'))
    return tf.nn.sigmoid(deconvolution(h_2, [batch_size,  s_h,    s_w,    1],name='g_dc2'))

def lrelu(x,leak=0.2):
    '''参考Rectier Nonlinearities Improve Neural Network Acoustic Models'''
    return tf.maximum(x,leak*x)  # 返回结果维度不变

def conv2d(input_,output_dim,name,k_h=5,k_w=5,s_h=2,s_w=2,stddev=0.02):
    '''普通的卷积层'''
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(stddev=stddev,shape=[k_h, k_w, input_.shape.as_list()[-1], output_dim]))
        conv = tf.nn.conv2d(input_,w, strides=[1,s_h,s_w, 1],padding='SAME')
        b = tf.Variable(tf.zeros([output_dim]))
        return tf.reshape(tf.nn.bias_add(conv,b),conv.shape)

def discriminator(x, x_gen):
    """

    :param x:   (784)
    :return:    (1); (1)
    """
    print(x)
    print(x_gen)
    x = tf.concat([x, x_gen], 0)

    h_cn1 = lrelu(conv2d(x, c_dim, name='d_conv_0'))
    print('dis_conv0  ' , h_cn1)
    h_cn2 = lrelu(conv2d(h_cn1, c_dim, name='d_conv_1_1'))
    print('dis_conv_1 ', h_cn2)
    h_cn3 = lrelu(conv2d(h_cn2, c_dim, name='d_conv_1_2'))
    print('dis_conv_2 ', h_cn3)

    h1 = lrelu(batch_norm(conv2d(h_cn3, df_dim, name='d_conv_2'), name='d_conv_2_bn1'))
    h1 = tf.reshape(h1, [batch_size + batch_size, -1])

    h2 = lrelu(batch_norm(linear(h1, dfc_dim, name='d_c_lear'), name='d_c_bn2'))

    h3 = linear(h2, 1, name='d_fc1')

    y_data = tf.nn.sigmoid(tf.slice(h3,          [0,0], [batch_size, -1], name=None))
    y_gen = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0],         [-1, -1], name=None))

    return y_data, y_gen


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


Z = tf.placeholder(tf.float32, shape=[None, 100], name='z')
# X = tf.placeholder(tf.float32, shape=[None, pic_size])
X = tf.placeholder(tf.float32,shape=[batch_size, img_height, img_width, 1], name='input_X')

G_sample = generator(Z) # 假图

print('gg_sample', G_sample)

print('X', X)

d_real, d_fake = discriminator(X, G_sample)

# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
d_loss = -tf.reduce_mean(tf.log(d_real + 1e-30) + tf.log(1. - d_fake + 1e-30)) # 不加这个1e-30会出现log(0)
g_loss = -tf.reduce_mean(tf.log(d_fake + 1e-30)) # tf有内置的sigmoid_cross_entropy_with_logits可以解决这个问题，但我没用它

# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

# 这一步很关键，主要是用来取出一切可以训练的参数，命名前缀决定了这个参数属于谁（建层的时候特地写的）
t_vars = tf.trainable_variables() # 所有可训练变量的列表
d_vars = [var for var in t_vars if var.name.startswith('d_')]
g_vars = [var for var in t_vars if var.name.startswith('g_')]

D_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars, global_step=global_step)
G_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars, global_step=global_step)


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=options)
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string)
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

image = read_and_decode('./tfrecords/train_3.tfrecord')
image_batch = tf.train.shuffle_batch([image], batch_size=mb_size, capacity = 50, min_after_dequeue=10)

checkpoint_dir = './tmp/ckpt_v2'

saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True,
                            gpu_options = gpu_options
                            )

config.gpu_options.visible_device_list = '3'

with tf.Session(config=config) as sess:

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('----------------restore pass ')
        ss = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('ckpt step == ', ss)
    else:
        print('ckpt step == NULL')
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

    print('-' * 20, checkpoint_dir)

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    for it in range(20000):

        if (it+1) % 100 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(mb_size, Z_dim)})
            samples = samples[:16]
            # print('len - ', len(samples))
            fig = plot(samples)
            plt.savefig('out_cnn_v2/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close()
            # print('save gen ')

        b_image = sess.run(image_batch)

        b_image = np.reshape(b_image, (-1, width, height, 1))

        _, D_loss_curr, step = sess.run([D_solver, d_loss, global_step], feed_dict={X: b_image, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, g_loss], feed_dict={X: b_image, Z: sample_Z(mb_size, Z_dim)})


        #每迭代2000次输出迭代数、生成器损失和判别器损失
        if it % 20 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
        if it % 200 == 0:
            saver.save(sess, checkpoint_dir+ '/model.ckpt', global_step=global_step)
            print('save ckpt', step)
        # print(b_image)
        # print(len(b_image))
        # i = 0
        # samples = b_image
        # fig = plot(samples)
        # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        # i += 1
        # plt.close()

    print('===============')
