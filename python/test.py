import tensorflow as tf
import numpy as np

def main(self, train=True):
    fileLists = np.genfromtxt(/home/wangguanshuo/lists/cropped/stn_cropped_with_params.npy, 
                              dtype=['S120', 'i8', 'i8', 'i8', 'i8', 'i8', 
                                     'f4', 'f4', 'f4', 'f4', 'f4', 'f4'], 
                              delimiter=' ')
    images = []
    xs = []
    hs = []
    ys = []
    ws = []
    labels = []
    if train:
        transforms = []
    for image, label, x, h, y, w, t1, t2, t3, t4, t5, t6 in fileLists:
        images.append(image)
        labels.append(label)
        xs.append(x)
        hs.append(h)
        ys.append(y)
        ws.append(w)
        if train:
            transforms.append([t1, t2, t3, t4, t5, t6])
    if train:
        image_path, x, h, y, w, tsf_params, label_index = tf.train.slice_input_producer(
                                                          [images, xs, hs, ys, ws, transforms, labels], 
                                                          shuffle=train)           
    else:
        image_path, x, h, y, w, label_index = tf.train.slice_input_producer([images, xs, hs, ys, ws, labels], shuffle=train)
        
    batches = []

    #image = tf.image.decode_jpeg(tf.read_file(image_path), channels=self.depth)
    #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #image = tf.image.crop_to_bounding_box(image, x, y, h, w)
    
    with tf.Session() as sess:
        print(xs.eval())