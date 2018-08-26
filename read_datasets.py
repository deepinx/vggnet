import tensorflow as tf
import numpy as np
import os


def get_file(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
 
    image_list = []
    label_list = []
    index = 0
    for sub_folders in os.listdir(file_dir):
        index += 1
        for image_name in os.listdir(os.path.join(file_dir, sub_folders)):
            if not image_name.endswith('.jpg') or image_name.startswith('.'):
                continue  # Skip!
            image_list.append(os.path.join(file_dir, sub_folders, image_name))
            label_list.append(index)
            # if image_name.startswith('cat.'):
            #     label_list.append(0)
            # else:
            #     label_list.append(1)
    print('There are %d images in the datasets' %(len(image_list)))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    return image_list, label_list

# def get_file(file_dir):
#     images = []
#     temp = []
#     for root, sub_folders, files in os.walk(file_dir):
#         for name in files:
#             # if not name.endswith('.jpg') or name.startswith('.'):
#             #     continue  # Skip!
#             images.append(os.path.join(root, name))
#         for name in sub_folders:
#             temp.append(os.path.join(root, name))
#     labels = []
#     for one_folder in temp:
#         n_img = len(os.listdir(one_folder))
#         letter = one_folder.split('/')[-1]
#         if letter == 'cat':
#             labels = np.append(labels, n_img * [0])
#         else:
#             labels = np.append(labels, n_img * [1])
#     # shuffle
#     temp = np.array([images, labels])
#     temp = temp.transpose()
#     np.random.shuffle(temp)
#     image_list = list(temp[:, 0])
#     label_list = list(temp[:, 1])
#     label_list = [int(float(i)) for i in label_list]

#     return image_list, label_list


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):

    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image,label],num_epochs=64)  #必须加入num_epochs参数，否则会有队列读写错误

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image = tf.image.per_image_standardization(image) # 将图片标准化
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch


def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels