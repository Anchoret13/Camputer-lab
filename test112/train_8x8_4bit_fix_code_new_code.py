'''
this code is used to compress raw data by 8*8 single layer compression;
Jianqiang Wang,NJU,8.12
'''
import numpy as np
import tensorflow as tf
import os
import math


###############set gpu###############
import keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

#####################################

input_image_size = 128

num_of_iter = 100
gradient_batch = 10
display_step = 200

starter_learning_rate = 0.0001
decay_steps = 25000
decay_rate = 0.75
global_step = tf.Variable(0, trainable=False)

NC = 1

block_size = 8
stride_size = 8
ncomp = 4
sc_int= 35
checkpoint_dir='./model_8x8_4bit_fix_code_new/'
graph_dir='./graph_8x8_4bit_fix_code_new/'

import h5py
f_train=h5py.File('./input_demosaiced_bggr.h5','r')
x_train=f_train['data'][:20000]#(24000, 128, 128, 1)

f_test=h5py.File('./input_demosaiced_bggr.h5','r')
x_test=f_test['data'][20000:]

f_code=h5py.File('./code_4bit_all_network.h5','r')
code=f_code['data'][:]

##########################use residual block as pre decompression model#################################

def residual_block(input_tense):
    tmp1 = tf.nn.relu(tf.layers.batch_normalization(
        tf.layers.conv2d(input_tense,64,[3,3],[1,1], padding='SAME')))
    tmp2 = tf.nn.relu(tf.layers.batch_normalization(
        tf.layers.conv2d(tmp1,16,[2,2],[1,1], padding='SAME')))
    tmp3 = tf.layers.batch_normalization(
        tf.layers.conv2d(tmp2,1,[1,1],[1,1], padding='SAME'))
    tmp4 = tmp3 + input_tense
    return tmp4


################################def mdoel################################################################
from xy_inception_tf_model import inception_module,inception_model

compW = tf.constant(code)
print("compW: \n" + str(compW))

with tf.name_scope('comp_Wf32'):
    compWf32 = tf.cast(compW,tf.float32)
    compWf32 = compWf32/sc_int
    print("\nBack to f32\n" + str(compWf32))

## input layer
with tf.name_scope('DATA'):
    Data_train = tf.placeholder(tf.float32,shape=[None,input_image_size,input_image_size,NC])
    print(Data_train)
    
with tf.name_scope('Compress_Conv'):
    comp = tf.nn.conv2d(Data_train, compWf32, strides=[1, stride_size, stride_size, 1], padding='VALID')
    print(comp)


#quantize feature
with tf.name_scope('Compress_Conv_mult'):
    compm = comp*45.
    print(compm)

with tf.name_scope('Compress_Conv_int'):
    compint = tf.cast(compm,tf.int8)
    print(compint)

with tf.name_scope('Compress_Conv_f32'):
    compf32 = tf.cast(compint,tf.float32)/45.
    print(compf32)

#######################no pre_decompression!###########################

with tf.name_scope('Pre_dcomp'):
    pdcomp = residual_block(compf32)
    print(pdcomp)
    
with tf.name_scope('Dcompress_Conv'):
    dcomp = tf.layers.conv2d_transpose(pdcomp,NC,[block_size,block_size],[stride_size,stride_size], padding='VALID')
    print(dcomp)


with tf.name_scope('inception_model'):
    outputs=inception_model(dcomp,8)


learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
Weight_ratio = tf.train.exponential_decay(0.9, global_step, 25000, 0.9, staircase=True)

with tf.name_scope('LOSS'):
    Loss = tf.reduce_mean(tf.square(outputs - Data_train))


def psnr(y_true, y_pred):
    mse = tf.reduce_mean(tf.reduce_mean(tf.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3,-2)))
    return 10 * tf.log(1. / mse) / tf.log(10.)
with tf.name_scope('PSNR'):
    Psnr = psnr(Data_train,outputs)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss,global_step = global_step)


writer = tf.summary.FileWriter(graph_dir)
init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)#restore the latest weights automatically
        print('Loading....................')
        test_loss = Loss.eval(feed_dict={Data_train:x_train[0:32,:,:,:]})
        print("Previous Loss: " + str(test_loss))
    else:
        sess.run(init)
    writer.add_graph(sess.graph)
    
    for i in range(global_step.eval(),num_of_iter+1):
        a = np.random.randint(0,19990,gradient_batch)
        optimizer.run(feed_dict={Data_train:x_train[a,:,:,:]})
        
        if i%display_step == 0:
            save_path = saver.save(sess,checkpoint_dir+'model_{}.ckpt'.format(i),global_step=global_step)
            #################metircs#####################################
            train_loss = Loss.eval(feed_dict={Data_train:x_train[a,:,:,:]})
            test_loss = Loss.eval(feed_dict={Data_train:x_test[0:32,:,:,:]})
            train_psnr = Psnr.eval(feed_dict={Data_train:x_train[a,:,:,:]})
            test_psnr = Psnr.eval(feed_dict={Data_train:x_test[0:32,:,:,:]})
            print("Iteration %d, learning rate %g, weight rate %g"%(
                i, learning_rate.eval(), Weight_ratio.eval()))
            print("Training loss %g,test loss %g,Training psnr %g,test psnr %g"%(train_loss,test_loss,train_psnr,test_psnr))
            co=compint.eval(feed_dict={Data_train:x_train[a,:,:,:]})
            print('mean:',np.mean(co))
            print('max:',np.max(co))
            print('min:',np.min(co))
