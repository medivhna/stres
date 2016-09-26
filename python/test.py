import tensorflow as tf
import numpy as np

def main():
    line_queue = tf.train.string_input_producer(['/home/wangguanshuo/lists/cropped/stn_cropped_with_params.npy'])
    reader = tf.TextLineReader()
    _, value = reader.read(line_queue)
    
    record_defaults = [[""], [0], [0], [0], [0], [0],
                       [.1], [.1],[.1],[.1],[.1],[.1]]
    filenames, labels, xs, hs, ys, ws, t1, t2, t3, t4, t5, t6 = tf.decode_csv(value, 
                                                                          record_defaults=record_defaults,
                                                                          field_delim=' ')
    image = tf.image.decode_jpeg(tf.read_file(filenames)) 
   
    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        
        # Retrieve a single instance:
        x, y, h, w = sess.run([xs, ys, hs, ws])
        print(x, y, h, w)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
