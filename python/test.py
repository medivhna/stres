import tensorflow as tf
import numpy as np

def main(self, train=True):
    line_queue = tf.train.string_input_producer([/home/wangguanshuo/lists/cropped/stn_cropped_with_params.npy])
    reader = tf.TextLineReader()
    _, value = reader.read(line_queue)
    
    record_defaults = [[""], [0], [0], [0], [0], [0],
                       [.1], [.1],[.1],[.1],[.1],[.1]]
    filenames, labels, x, h, y, w, t1, t2, t3, t4, t5, t6 = 
                      tf.decode_csv(value, record_defaults=record_defaults)
    
    print(filenames)