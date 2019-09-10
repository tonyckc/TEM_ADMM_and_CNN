import datetime
import numpy as np
import sys
sys.path.insert(0, './src')
import tensorflow as tf
import os
from PIL import Image
import scipy.misc as misc
import random
from utils import create_save_folder, get_img, get_next_batch
from argparse import ArgumentParser
import model as net

def train(input_data, real_data, test_data, training_id, MODEL_SAVE_PATH, TENSORBOARD_SAVE_PATH,
          DENOISING_IMG_PATH, LEARNING_RATE_BASE, LEARNING_RATE_DECAY, EPOCHS, LOAD_MODEL = False, BATCH_SIZE = None,
          IMG_SIZE = None, DEVICE = None):
    global_steps = tf.Variable(initial_value=0, name=
                              'global_steps', trainable=False)
    input_placeholder = tf.placeholder(dtype=tf.float32
                                       , shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input_placehloder')
    real_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3], name='real_placeholder')
    with tf.device(DEVICE):
            net_output = net.denoiser(input_placeholder,reuse=False)
            net_output_test = net.denoiser(input_placeholder,reuse=True)
            with tf.name_scope('losses'):
                 losses = tf.losses.mean_squared_error(real_placeholder, net_output)
                 tf.summary.scalar('loss',losses)
            learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_steps,
                (len(input_data)//BATCH_SIZE),
                LEARNING_RATE_DECAY,
                staircase=True
            )
            saver = tf.train.Saver()
            trainer = tf.train.AdamOptimizer(learning_rate).minimize(losses,global_steps)

            merged = tf.summary.merge_all()

            config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                writer = tf.summary.FileWriter(TENSORBOARD_SAVE_PATH, sess.graph)
                start = 0
                if LOAD_MODEL:
                     print("Reading checkpoints...")
                     ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                     if ckpt and ckpt.model_checkpoint_path:
                         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                         saver.restore(sess, os.path.join(MODEL_SAVE_PATH,ckpt_name))
                         global_steps = ckpt.model_checkpoint_path.split('/')[-1]\
                                                                            .split('-')[-1]
                         print("Loading success,global_step is % s" % (global_steps))
                     start = int(global_steps)
                steps = start

                print('Start Training....\n')
                for epoch in range(EPOCHS):
                    rand_list = random.sample(range(0, len(input_data)//BATCH_SIZE),len(input_data)//BATCH_SIZE)
                    for num in range(len(input_data)//BATCH_SIZE):
                        rand_point = rand_list[num]

                        train_ = sess.run(trainer, feed_dict={input_placeholder: get_next_batch(input_data,
                             rand_point, BATCH_SIZE), real_placeholder: get_next_batch(real_data, rand_point, BATCH_SIZE)})
                        print("Epoch: %d, Iteration: %d, At: %s" % (
                        epoch, num, datetime.datetime.now()))
                        if num % 5 == 0:
                            # five iterations write the summary and output the losses
                            summary, loss = sess.run([merged, losses], feed_dict={input_placeholder: get_next_batch(input_data,
                             rand_point, BATCH_SIZE), real_placeholder: get_next_batch(real_data, rand_point, BATCH_SIZE)})
                            writer.add_summary(summary, global_step=steps)

                            print("Epoch: %d, Iteration: %d, Loss: %d, At: %s" % (epoch, num,loss,datetime.datetime.now()))
                        if num % 200 == 0:
                            print("Start test.....\n")
                            test = sess.run(net_output_test, feed_dict={input_placeholder: get_next_batch(test_data, 0, BATCH_SIZE)})
                            img_id = 0
                            for img in test[1:10,:,:,:]:

                                img_ = np.reshape(img, (IMG_SIZE, IMG_SIZE, -1)).astype(np.uint8)
                                img_ = Image.fromarray(img_)
                                save_path = DENOISING_IMG_PATH + '//{}_{}_{}.png'.format(training_id, steps, img_id)
                                img_.save(save_path)
                                img_id += 1
                            print('Test Images are Saved\n')
                        if num % 200 == 0:
                            path = MODEL_SAVE_PATH + os.path.sep + 'model.ckpt'
                            saver.save(sess, path, global_step=steps)
                    steps += 1