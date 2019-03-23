'''
this code is used to compress raw data by 8*8 single layer compression;
i want to use this to learn the best compression code.
Jianqiang Wang,NJU,8.17
'''
import numpy as np
import tensorflow as tf
import os
import math


###############set gpu###############
import keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

#####################################

input_image_size = 128

num_of_iter = 10000
gradient_batch = 100
display_step = 200

starter_learning_rate = 0.0002
decay_steps = 25000
decay_rate = 0.75
global_step = tf.Variable(0, trainable=False)

NC = 1

block_size = 8
stride_size = 8
ncomp = 4
#sc_int=20 
checkpoint_dir='./model_8x8_learn_code_no_quantization_v7/'
graph_dir='./graph_8x8_learn_code_no_quantization_v7/'

import h5py
f_train=h5py.File('./input_demosaiced_bggr.h5','r')
x_train=f_train['data'][:20000]#(24000, 128, 128, 1)

f_test=h5py.File('./input_demosaiced_bggr.h5','r')
x_test=f_test['data'][:]

#f_code=h5py.File('./code3.h5','r')
f_code=h5py.File('./code_binary.h5','r')
code=f_code['data'][:]

##########################use residual block as pre decompression model#################################
import h5py
def write_hdf5(x,filename):
    with h5py.File(filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)

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
'''
compW = tf.get_variable("comp_W", [block_size, block_size, NC, ncomp], initializer=tf.keras.initializers.glorot_uniform(seed=7))
print("compW: \n" + str(compW))
'''
compW = tf.constant(code)
print("compW: \n" + str(compW))
'''
with tf.name_scope('comp_Wf32'):
    compWf32 = tf.cast(compW,tf.float32)
    compWf32 = compWf32/sc_int
    print("\nBack to f32\n" + str(compWf32))
'''
## input layer
with tf.name_scope('DATA'):
    Data_train = tf.placeholder(tf.float32,shape=[None,input_image_size,input_image_size,NC])
    print(Data_train)
    
with tf.name_scope('Compress_Conv'):
    comp = tf.nn.conv2d(Data_train, compW, strides=[1, stride_size, stride_size, 1], padding='VALID')
    print(comp)


#quantize feature
with tf.name_scope('Compress_Conv_mult'):
    compm = comp*6.
    print(compm)

with tf.name_scope('Compress_Conv_int'):
    compint = tf.cast(compm,tf.uint8)
    print(compint)

with tf.name_scope('Compress_Conv_f32'):
    compf32 = tf.cast(compint,tf.float32)/6.
    print(compf32)

#######################no pre_decompression!###########################
'''
with tf.name_scope('Pre_dcomp'):
    pdcomp = residual_block(compf32)
    print(pdcomp)
'''  
with tf.name_scope('Dcompress_Conv'):
    outputs = tf.layers.conv2d_transpose(compf32,NC,[block_size,block_size],[stride_size,stride_size], padding='VALID')
    print(outputs)

'''
with tf.name_scope('inception_model'):
    outputs=inception_model(dcomp,8)
'''

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
        code = compW.eval()
        #write_hdf5(code,'code_init.h5')
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

    code = compW.eval()
    #write_hdf5(code,'code_learned_without_quantization.h5')



#############################predict########################################
#Get total dip in 100 events each time using GPU
ss=500
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
    co=comp.eval(feed_dict={Data_train:x_test[0:ss,:,:,:]})

Dip = dip
Co = co
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)#restore the latest weights automatically
    else:
        pass
    #saver.restore(sess,'./model_quantlization/model_200000.ckpt-200001')
    for i in range(1,ns):
        tmp=sess.run(outputs,feed_dict={Data_train:x_test[(i*ss):(i*ss+ss),...]})
        tmp2=comp.eval(feed_dict={Data_train:x_test[(i*ss):(i*ss+ss),...]})
        Dip=np.concatenate((Dip,tmp),axis=0)
        Co=np.concatenate((Co,tmp2),axis=0)
        

mse_tot=np.mean(np.square(x_test-Dip))
print(mse_tot)
print('mean:',np.mean(Co))
print('max:',np.max(Co))
print('min:',np.min(Co))

import h5py
def write_hdf5(x,filename):
    with h5py.File(filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)

write_hdf5(Co,'Co_learned_binary.h5')

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

