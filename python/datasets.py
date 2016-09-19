# encoding: utf-8

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

_ALIGN_KEYPOINT = [[0.2, 0.8, 0.5], [0.25, 0.25, 0.9]]

class DataSet:
    """Dataset Class."""
    def __init__(self, list_path, batch_size, height, width, depth, delimeter, num_preprocess_threads=8):
        """Dataset constructor."""

        self.list_path = list_path
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth = depth
        self.delimeter = delimeter
        self.num_preprocess_threads = num_preprocess_threads

    def distort_color(self, image, thread_id=0):
        """Distort the color of the image.

        Each color distortion is non-commutative and thus ordering of the color ops
        matters. Ideally we would randomly permute the ordering of the color ops.
        Rather then adding that level of complication, we select a distinct ordering
        of color ops for each preprocessing thread.

        Args:
          image: Tensor containing single image.
          thread_id: preprocessing thread ID.
          scope: Optional scope for op_scope.
        Returns:
          color-distorted image
        """
        with tf.op_scope([image], 'distort_color'):
            color_ordering = thread_id % 2

            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)

            # The random_* ops do not necessarily clamp.
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image
    def distort_image(self, image, height, width, bbox, thread_id=0, scope=None):
        """Distort one image for training a network.

      Distorting images provides a useful technique for augmenting the data
      set during training in order to make the network invariant to aspects
      of the image that do not effect the label.

      Args:
        image: 3-D float Tensor of image
        height: integer
        width: integer
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged
          as [ymin, xmin, ymax, xmax].
        thread_id: integer indicating the preprocessing thread.
        scope: Optional scope for op_scope.
      Returns:
        3-D float Tensor of distorted image used for training.
      """
        with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):

            # NOTE(ry) I unceremoniously removed all the bounding box code.
            # Original here: https://github.com/tensorflow/models/blob/148a15fb043dacdd1595eb4c5267705fbd362c6a/inception/inception/image_processing.py

            distorted_image = image

            # This resizing operation may distort the images because the aspect
            # ratio is not respected. We select a resize method in a round robin
            # fashion based on the thread number.
            # Note that ResizeMethod contains 4 enumerated resizing methods.
            resize_method = thread_id % 4
            distorted_image = tf.image.resize_images(distorted_image, height,
                                                     width, resize_method)
            # Restore the shape since the dynamic slice based upon the bbox_size loses
            # the third dimension.
            distorted_image.set_shape([height, width, 3])

            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Randomly distort the colors.
            distorted_image = self.distort_color(distorted_image, thread_id)

            return distorted_image

    def eval_image(self, image, height, width, scope=None):
        """Prepare one image for evaluation.

      Args:
        image: 3-D float Tensor
        height: integer
        width: integer
        scope: Optional scope for op_scope.
      Returns:
        3-D float Tensor of prepared image.
      """
        with tf.op_scope([image, height, width], scope, 'eval_image'):
            # Resize the image to the original height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
            return image

    def cnt_samples(self, filepath):
        return sum(1 for line in open(filepath))

    def inputs(self, train=True):
        fileLists = np.genfromtxt(self.list_path, 
                                 dtype=['S120', 'i8'], delimiter=self.delimeter)
        images = []
        labels = []
        for image, label in fileLists:
            images.append(image)
            labels.append(label)
        
        filename, label_index = tf.train.slice_input_producer([images, labels], shuffle=train)
            
        images_and_labels = []

        for tid in range(self.num_preprocess_threads):
            image = tf.image.decode_jpeg(tf.read_file(filename), channels=self.depth)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            
            if train:
                image = self.distort_image(image, self.height, self.width, bbox=[], thread_id=tid)
            else:
                image = self.eval_image(image, self.height, self.width)

            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=self.batch_size,
            capacity=2 * self.num_preprocess_threads * self.batch_size)

        images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, self.depth])

        return images, tf.reshape(label_index_batch, [self.batch_size])

    def inputs2(self, train=True):
        fileLists = np.genfromtxt(self.list_path, 
                                 dtype=['S120', 'i8'], delimiter=self.delimeter)
        images = []
        rects = []
        labels = []
        if train:
            transforms = []
        for image, label in fileLists:
            images.append(image)
            rects.append(image[1:-4]+'.txt')
            labels.append(label)
            if train:
                transforms.append(image[1:-4]+'.tsf')
        
        if train:
            image_path, rect_path, tsf_path, label_index = tf.train.slice_input_producer([images, rects, transforms, labels], shuffle=train)
        else:
            image_path, rect_path, label_index = tf.train.slice_input_producer([images, rects, labels], shuffle=train)
            
        batches = []

        for tid in range(self.num_preprocess_threads):
            with open(tf.read_file(rect_path)) as rct:
                rect_line = rct.split()
                [x, y, h, w] = map(int, rect_line[0:3])
            image = tf.image.decode_jpeg(tf.read_file(image_path), channels=self.depth)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.crop_to_bounding_box(image, x, y, h, w)
            
            if train:
                image = self.distort_image(image, self.height, self.width, bbox=[], thread_id=tid)
                with open(tf.read_file(tsf_path)) as tsf:
                    tsf_line = tsf.split()
                    transform = map(float, tsf_line)
                batches.append([image, label_index, transform])
            else:
                image = self.eval_image(image, self.height, self.width)
                batches.append([image, label_index])
        
        if train:
            images, label_index_batch, transforms = tf.train.batch_join(
                batches,
                batch_size=self.batch_size,
                capacity=2 * self.num_preprocess_threads * self.batch_size)

            images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, self.depth])

            return images, tf.reshape(label_index_batch, [self.batch_size]), tf.reshape(transforms, [self.batch_size, 6])
        else:
            images, label_index_batch = tf.train.batch_join(
                batches,
                batch_size=self.batch_size,
                capacity=2 * self.num_preprocess_threads * self.batch_size)

            images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, self.depth])

            return images, tf.reshape(label_index_batch, [self.batch_size])
