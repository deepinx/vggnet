#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 00:12:41 2017
@author: hjxu
"""

import tensorflow as tf
import numpy as np
import os
# import matplotlib.pyplot as plt
from PIL import Image

 
#%%

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
 
    image_list = []
    label_list = []
    for image_name in os.listdir(file_dir):
        if not image_name.endswith('.jpg') or image_name.startswith('.'):
            continue  # Skip!
        image_list.append(os.path.join(file_dir, image_name))
        if image_name.startswith('cat.'):
            label_list.append(0)
        else:
            label_list.append(1)
    print('There are %d images in the datasets' %(len(image_list)))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    return image_list, label_list
 
# def get_files(file_dir):
#     '''
#     Args:
#         file_dir: file directory
#     Returns:
#         list of images and labels
#     '''
 
#     image_list = []
#     label_list = []
#     category = -1;
#     for sub_folder in os.listdir(file_dir):
#         category += 1;
#         for image_name in os.listdir(os.path.join(file_dir, sub_folder)):
#             if not image_name.endswith('.jpg') or image_name.startswith('.'):
#                 continue  # Skip!
#             image_list.append(os.path.join(file_dir, sub_folder, image_name))
#             label_list.append(category)
#     print('There are %d images in the datasets' %(len(image_list)))
    
#     temp = np.array([image_list, label_list])
#     temp = temp.transpose()
#     np.random.shuffle(temp)
    
#     image_list = list(temp[:, 0])
#     label_list = list(temp[:, 1])
#     label_list = [int(i) for i in label_list]
    
#     return image_list, label_list
 
#%%

 
def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = (save_dir + name)
    n_samples = len(labels)
    
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('Transform start......')
    for i in np.arange(0, n_samples):
        try:
            image = Image.open(images[i])
            image= image.resize((224,224))
            image_raw = image.tobytes()              #将图片转化为原生bytes
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
    
 
#%%
 
def read_and_decode(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
    
    image = tf.reshape(image, [224, 224,3])
    label = tf.cast(img_features['label'], tf.int64)    
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = 2000)
    return image_batch, tf.reshape(label_batch, [batch_size])


if __name__ == '__main__':
    save_dir = './'
    name = 'train.tfrecords'
    image_list, label_list = get_files("./train")
    convert_to_tfrecord(image_list, label_list, save_dir, name)