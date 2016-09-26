import tensorflow as tf
import numpy as np

def main():
    train = True
    line_queue = tf.train.string_input_producer(['/home/wangguanshuo/lists/cropped/stn_cropped_with_params.npy'])
    reader = tf.TextLineReader()
    _, value = reader.read(line_queue)
    record_defaults = [[""], [0], [0], [0], [0], [0],
                       [.1], [.1],[.1],[.1],[.1],[.1]]
    filenames, labels, x, h, y, w, t1, t2, t3, t4, t5, t6 = tf.decode_csv(value, 
                                                                          record_defaults=record_defaults,
                                                                          field_delim=' ')    
    batches = []
    for tid in range(4):
        image = tf.image.decode_jpeg(tf.read_file(filenames), channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.crop_to_bounding_box(image, x, y, h, w)
        tsf_params = [t1, t2, t3, t4, t5, t6]
        
        batches.append([image, labels, tsf_params])
    
    images, label_index_batch, transforms = tf.train.shuffle_batch_join(
            batches,
            batch_size=128,
            capacity=2 * 4 * 128,
            min_after_dequeue = 2 * 128) 
    print(images, label_index_batch, transforms)
    # with tf.Session() as sess:
        # # Start populating the filename queue.
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        
        # # Retrieve a single instance:
        # x, y, h, w = sess.run([xs, ys, hs, ws])
        # print(x, y, h, w)

        # coord.request_stop()
        # coord.join(threads)

if __name__ == '__main__':
    main()
