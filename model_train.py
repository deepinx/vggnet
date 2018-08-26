import numpy as np
import tensorflow as tf
import VGG16_model as model
# from create_tfrecords import *
import read_datasets as reader2

img_width = 224
img_height = 224

if __name__ == '__main__':

    X_train, y_train = reader2.get_file("D:/Image_Processing/Datasets/cub_200_2011/images/")
    image_batch, label_batch = reader2.get_batch(X_train, y_train, img_width, img_height, 25, 256)
    # image_batch, label_batch = read_and_decode("./train.tfrecords", batch_size=25)

    x_imgs = tf.placeholder(tf.float32, [None, img_width, img_height, 3])
    y_imgs = tf.placeholder(tf.int32, [None, model.class_num])

    vgg = model.vgg16(x_imgs)
    y_vgg_fc8 = vgg.probs
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_vgg_fc8, labels=y_imgs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())   # tf.train.string_input_producer need this initializer
        vgg.load_weights('./vgg16_weights.npz', sess)
        saver = vgg.saver()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        import time
        start_time = time.time()
        for i in range(1000):
            if coord.should_stop(): 
                break
            image, label = sess.run([image_batch, label_batch])
            label = (np.arange(model.class_num) == label[:,None]).astype(np.float32)

            sess.run(optimizer, feed_dict={x_imgs: image, y_imgs: label})
            loss_record = sess.run(loss, feed_dict={x_imgs: image, y_imgs: label})
            print("now the loss is %f " % loss_record)
            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time
            print("----------step %d is finished---------------" % i)

            if i % 200 == 0 and i != 0:
                saver.save(sess, 'checkpoint/%05d.ckpt' % i)
                print('save model to checkpoint/%05d.ckpt' % i)

        coord.request_stop()
        coord.join(threads)
        print("Optimization Finished!")

    # i = 0
    # try:
    #     while not coord.should_stop():
    #         image, label = sess.run([image_batch, label_batch])
    #         label = (np.arange(class_num) == label[:,None]).astype(np.float32)
    #         sess.run(optimizer, feed_dict={x_imgs: image, y_imgs: label})
    #         loss_record = sess.run(loss, feed_dict={x_imgs: image, y_imgs: label})
    #         print("step %d the loss is %f " % (i, loss_record))
    #         i += 1
    # except tf.errors.OutOfRangeError:
    #     print('Done training')
    # finally:
    #     coord.request_stop()
    # coord.join(threads)
    # sess.close()