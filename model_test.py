import tensorflow as tf
from scipy.misc import imread, imresize
import VGG16_model as model

imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
sess = tf.Session()
vgg = model.vgg16(imgs)
fc3_cat_and_dog = vgg.probs
saver = vgg.saver()
saver.restore(sess, './model/')

import os
for root, sub_folders, files in os.walk('./test/'):
    i = 0
    cat = 0
    dog = 0
    for name in files:
        i += 1
        filepath = os.path.join(root, name)

        try:
            img1 = imread(filepath, mode='RGB')
            img1 = imresize(img1, (224, 224))
        except:
            print("remove", filepath)

        prob = sess.run(fc3_cat_and_dog, feed_dict={vgg.imgs: [img1]})
        import numpy as np
        max_index = np.argmax(prob)
        if max_index == 0:
            cat += 1
        else:
            dog += 1
        if i % 50 == 0:
            acc = (cat * 1.)/(dog + cat)
            print(acc)
            print("-----------img number is %d------------" % i)