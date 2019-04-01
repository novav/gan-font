import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def variable_init(size):
    in_dims = size[0]
    w_stddev = 1./ tf.sqrt(in_dims/2.)
    return tf.random_normal(shape=size, stddev = w_stddev)


# 28x28 = 784

# def net_build():
X = tf.placeholder(tf.float32, shape=[None, 784])
D_W1 = tf.Variable(variable_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(variable_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(variable_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(variable_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))
theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    """

    :param z:   (100)
    :return:    (784)
    """
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)

    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2

    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def discriminator(x):
    """

    :param x:   (784)
    :return:    (1); (1)
    """
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)

    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig




G_sample = generator(Z)

D_real, D_logit_real = discriminator(X)
D_Fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                     labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)
))

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)
))


D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100

mnist= input_data.read_data_sets('./data/MNIST/', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
if not os.path.exists('./out/'):
    os.mkdir('./out')

i = 0
for it in range(20000):

    if it % 20 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        print(len(samples))
        print(samples)
        fig = plot(samples)
        exit()
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close()

    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})


    #每迭代2000次输出迭代数、生成器损失和判别器损失
    if it % 2000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()