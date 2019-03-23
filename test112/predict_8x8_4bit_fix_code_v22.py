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

####################changeable parameters###############################

input_image_size = 128

NC = 1

block_size = 8
stride_size = 8
ncomp = 4
sc_int= 20
checkpoint_dir='./model_8x8_4bit_fix_code_v2/'
graph_dir='./graph_8x8_4bit_fix_code_v2/'

import h5py
f_train=h5py.File('./input_demosaiced_bggr.h5','r')
#x_train=f_train['data'][:20000]#(24000, 128, 128, 1)

f_test=h5py.File('./input_demosaiced_bggr.h5','r')
x_test=f_test['data'][20000:]

f_code=h5py.File('./code3.h5','r')
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
    compm = comp*50.
    print(compm)

with tf.name_scope('Compress_Conv_int'):
    compint = tf.cast(compm,tf.uint8)
    print(compint)

with tf.name_scope('Compress_Conv_f32'):
    compf32 = tf.cast(compint,tf.float32)/50.
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
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)#restore the latest weights automatically
    else:
        pass
    #saver.restore(sess,'./model_quantlization/model_200000.ckpt-200001')
    dip=sess.run(outputs,feed_dict={Data_train:x_test[0:ss,:,:,:]})

Dip = dip
with tf.Session() as sess:
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
import h5py
def write_hdf5(x,filename):
    with h5py.File(filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)

write_hdf5(Dip,'raw_output_fix_code_4bit_4000.h5')
#write_hdf5(x_test,'raw_label.h5')

