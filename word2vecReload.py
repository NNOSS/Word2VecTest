from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import re

from tensorflow.contrib.tensorboard.plugins import projector

filename = "./all_data.txt" #maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with open(filename, 'rb') as f:
    data = []
    pattern = re.compile("^\w\S+")
    #m = pattern.match(f.read())#.split()
    


    for line in f.readlines():
      m = pattern.match(line)
      if (m != None):
        #print(m.group(0))
        data.append(tf.compat.as_str(m.group(0)).lower())
    #data = tf.compat.as_str(f.read()).split()
  return data
  """with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data"""


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2  # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.
learning_rate = 1.0 # 1 seems high, but it was the default value in this code

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


sess = tf.Session()

saver = tf.train.import_meta_graph('./model')
saver.restore(sess,tf.train.latest_checkpoint('./Model2/'))

graph = tf.get_default_graph()
#namelist = [n.name for n in tf.get_default_graph().as_graph_def().node]
#print(namelist)
decoder_in = graph.get_tensor_by_name("enc_net/hidden_encode/Relu:0")
dec_final = graph.get_tensor_by_name("decoding:0")
#f_255 = tf.image.convert_image_dtype (f, dtype=tf.uint8)
#dec_final = tf.summary.image('dec_test', f_255, max_outputs = 1)

step = 0
merged = tf.summary.merge_all()
sum_writer = tf.summary.FileWriter("logs_enctest", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

while(True):
	in_ind = raw_input("pick a valid integer: ")
	try:
		in_ind = int(in_ind)
	except:
		continue
	test = np.zeros(50)

	test[in_ind] = 10;
	test = np.expand_dims(test, axis=0)
	feed_dict = {decoder_in: test}
	out = sess.run(dec_final,feed_dict)
	print(out)
	sum_writer.add_summary(out, step)
	step += 1;