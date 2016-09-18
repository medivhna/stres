# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet feature extractor module.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
from datetime import datetime

import numpy as np
from scipy import io
import tensorflow as tf

import resnet_model as model # change the import to change the model
from datasets import DataSet

# Extracted Data Settings
tf.app.flags.DEFINE_integer('batch_size', 100, 'the number of images in a batch.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. ')

tf.app.flags.DEFINE_integer('image_height', 128, 'Image height.')
tf.app.flags.DEFINE_integer('image_width', 128, 'Image width.')
tf.app.flags.DEFINE_integer('image_depth', 3, 'Image depth.')
tf.app.flags.DEFINE_integer('num_examples', 5000000, "Number of examples for training")
tf.app.flags.DEFINE_integer('num_classes', 10000, "Number of classes")

# Path Settings for extracting
tf.app.flags.DEFINE_string('delimeter', ' ', "Delimeter of the list")
tf.app.flags.DEFINE_string('fea_dir', '', 'Directory to keep feature files.')
tf.app.flags.DEFINE_string('fea_file', '', 'name of the feature file.')
tf.app.flags.DEFINE_string('extract_list_path', '', 'Filename for training data list.')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'Path to the model checkpoint')

# Network Settings
tf.app.flags.DEFINE_integer('num_layers', 56, 'Number of network layers(num_layers-2 must be diviced by 9)')


FLAGS = tf.app.flags.FLAGS

def extract(hps):
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    extract_set = DataSet(list_path=FLAGS.extract_list_path, 
                          batch_size=FLAGS.batch_size, 
                          height=FLAGS.image_height, 
                          width=FLAGS.image_width, 
                          depth=FLAGS.image_depth,
                          delimeter=FLAGS.delimeter,
                          num_preprocess_threads=1)
    images, labels = extract_set.inputs(train=False)

    net = model.Network(hps, images, labels, train=False)
    net.inference()
    saver = tf.train.Saver()

    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(sess, os.path.join(FLAGS.checkpoint_path,
                              ckpt.model_checkpoint_path))

          # Assuming model_checkpoint_path looks something like:
          #   /my-favorite-path/imagenet_train/model.ckpt-0,
          # extract global_step from it.
          global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
          print('Succesfully loaded model from %s at step=%s.' %
                (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found')
        return

      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        print('%s: starting feature extraction...' % (datetime.now()))

        start_time = time.time()
        step = 0
        while step < num_iter and not coord.should_stop():
          [fea0, labels0] = sess.run([net.logits, labels])

          if step == 0:
            fea = fea0
            labels_output = labels0
          else:
            fea = np.row_stack((fea, fea0))
            labels_output = np.append(labels_output, labels0)

          step += 1

          if step % 10 == 0:
            duration = time.time() - start_time
            sec_per_batch = duration / 10.0
            examples_per_sec = FLAGS.batch_size / sec_per_batch
            print('%s: [%d batches extraction done of %d] (%.1f examples/sec; %.3f'
                  'sec/batch)' % (datetime.now(), step, num_iter,
                                  examples_per_sec, sec_per_batch))
            start_time = time.time()
          
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  return fea[0:FLAGS.num_examples, :], labels_output[0:FLAGS.num_examples]

def main(argv=None):
  hps = model.HParams(batch_size=FLAGS.batch_size,
                      num_classes=FLAGS.num_classes,
                      num_gpus=FLAGS.num_gpus,
                      initial_learning_rate=0.0,
                      lr_decay_steps=0.0,
                      lr_decay_factor=0.0,
                      optimizer='',
                      num_layers=FLAGS.num_layers,
                      use_bottleneck=True,
                      weight_decay_rate=0.0001,
                      relu_leakiness=0)

  if not tf.gfile.Exists(FLAGS.fea_dir):
    tf.gfile.MakeDirs(FLAGS.fea_dir)

  features, labels = extract(hps)
  io.savemat(os.path.join(FLAGS.fea_dir, FLAGS.fea_file), {'wfea': features, 'labels': labels})
  print('Features have saved to feature_dir.')
  print('Done.')

if __name__ == '__main__':
  tf.app.run()

