'''
this code is  used to test the performance of model on patch
Jianqiang Wang,NJU,8.4
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
import os
import math
np.random.seed(73)
from skimage.measure import compare_ssim,compare_psnr


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True


####################changeable parameters###############################

input_image_size = 128

num_of_iter = 100000
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
sc_int= 900 #1024.
checkpoint_dir='./model_8x8_8bit/'
graph_dir='./graph_8x8_8bit/'

import h5py

f_test=h5py.File('./demosaiced_reconstruct_fix_code_binary.h5','r')
x_test=f_test['data'][5000:]


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
################################def mdoel################################################################
from xy_inception_tf_model import inception_module,inception_model

compW = tf.get_variable("comp_W", [block_size, block_size, NC, ncomp], initializer=tf.keras.initializers.glorot_uniform(seed=7))
print("compW: \n" + str(compW))

with tf.name_scope('comp_Wi'):
    compWi = compW*sc_int
    compWi = tf.cast(compWi,tf.int8)#1st constraint;max is 127
    print("\limit it an 8 bit integer\n" + str(compWi))

with tf.name_scope('comp_Wf32'):
    compWf32 = tf.cast(compWi,tf.float32)
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
    compm = comp*127.
    print(compm)

with tf.name_scope('Compress_Conv_int'):
    compint = tf.cast(compm,tf.int8)
    print(compint)

with tf.name_scope('Compress_Conv_f32'):
    compf32 = tf.cast(compint,tf.float32)/127.
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



#########define loss and load weights######################################
with tf.name_scope('LOSS'):
    Loss = tf.reduce_mean(tf.square(outputs - Data_train))

writer = tf.summary.FileWriter(graph_dir)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#############################predict########################################
#Get total dip in 100 events each time using GPU
ss=100
ns=int(x_test.shape[0]/ss)
print("number of steps= " + str(ns) + " , step size= " + str(ss))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)#restore the latest weights automatically
    else:
        pass
    #saver.restore(sess,'./model_quantlization/model_200000.ckpt-200001')
    dip=sess.run(outputs,feed_dict={Data_train:x_test[0:ss,:,:,:]})

Dip = dip
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)#restore the latest weights automatically
    else:
        pass
    #saver.restore(sess,'./model_quantlization/model_200000.ckpt-200001')
    for i in range(1,ns):
        tmp=sess.run(outputs,feed_dict={Data_train:x_test[(i*ss):(i*ss+ss),...]})
        Dip=np.concatenate((Dip,tmp),axis=0)

mse_tot=np.mean(np.square(x_test-Dip))
print(mse_tot)


##################################calculate metrics##############################
###according to Chao's experiments, if you scale by 2^14-1, psnr will be very small(around 7dbB)
output=(Dip*255).astype('uint8')
label=(x_test*255).astype('uint8')

from skimage.measure import compare_ssim,compare_psnr
avg_psnr = np.zeros([1])
avg_ssim = np.zeros([1])
for i in range(output.shape[0]):
    psnr =compare_psnr(output[i,:,:,0], label[i,:,:,0])
    ssim =compare_ssim(output[i,:,:,0], label[i,:,:,0], multichannel=False)  

    avg_ssim[0] += ssim
    avg_psnr[0] += psnr
   
    #print and save
avg_psnr = avg_psnr/output.shape[0]
avg_ssim = avg_ssim/output.shape[0]

print('avg_psnr:',avg_psnr,'\n','avg_ssim:',avg_ssim)


'''
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 18.0) 

number=4
ids=np.random.randint(0,500,number)

plt.figure(figsize=(9, 18))
for i in range(number):
    idv=ids[i]
    plt.subplot(number,2,i*2+1)
    plt.imshow(Dip[idv,:,:,0],cmap='gray')
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)

    plt.subplot(number,2,i*2+2)    
    plt.imshow(x_test[idv,:,:,0],cmap='gray')
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)

plt.show()

'''
'''
import h5py
def write_hdf5(x,filename):
    with h5py.File(filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)

write_hdf5(Dip,'raw_output.h5')
write_hdf5(x_test,'raw_label.h5')
'''
