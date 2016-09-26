import tensorflow as tf
import numpy as np

def main():
    line_queue = tf.train.string_input_producer([self.list_path])
    reader = tf.TextLineReader()
    _, value = reader.read(line_queue)
    record_defaults = [[""], [0], [0], [0], [0], [0],
                       [.1], [.1],[.1],[.1],[.1],[.1]]
    filenames, labels, x, h, y, w, t1, t2, t3, t4, t5, t6 = tf.decode_csv(value, 
                                                                          record_defaults=record_defaults,
                                                                          field_delim=' ')    

    for tid in range(self.num_preprocess_threads):
        image = tf.image.decode_jpeg(tf.read_file(filenames), channels=self.depth)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.crop_to_bounding_box(image, x, y, h, w)
        tsf_params = tf.pack([t1, t2, t3, t4, t5, t6])
        
        if train:
            image = self.distort_image(image, self.height, self.width, bbox=[], thread_id=tid)
            batches.append([image, labels, tsf_params])
        else:
            image = self.eval_image(image, self.height, self.width)
            batches.append([image, labels])
    
    print(batches)
   
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
