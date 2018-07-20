# To train a 300d embedding for the charactors in 
import tensorflow as tf
import numpy as np
import pandas as pd
from Input import InputData 
from tqdm import tqdm




# use NCE https://mk-minchul.github.io/NCE/
# how to use optimizer https://gist.github.com/DominicBreuker/c1082d02456c4186c1a5f77e12972b85
class Word2Vec(object):
    def __init__(self, data_path, embedding_size):
        
        self.data = InputData(data_path, 10000)
        self.batch_size = 1024
        self.vocab_size = self.data.word_count
        self.num_sampled = 5
        self.window_size = 5
        self.learning_rate = 0.025
        
        
        center_words = tf.placeholder(tf.int32, shape = [self.batch_size])
        target_words = tf.placeholder(tf.int32, shape= [self.batch_size, 1])

        embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embed_matrix, center_words)
        
        nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size], stddev=1.0 / embedding_size ** 0.5))
        nce_bias = tf.Variable(tf.zeros([self.vocab_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias,
                                             labels=target_words,
                                             inputs=embed,
                                             num_sampled=self.num_sampled,
                                             num_classes=self.vocab_size))
        
  
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.center_words_placeholder = center_words
        self.target_words_placeholder = target_words
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
        self.loss = loss
        
        norm = tf.sqrt(tf.reduce_sum(tf.square(embed_matrix), 1, keepdims=True))
        self.normalized_embeddings = embed_matrix / norm
        
  
    def train_epoch(self, sess):
        pos_pairs = self.data.get_batch_pairs(self.batch_size, self.window_size)
        center_words = [pair[0] for pair in pos_pairs]
        target_words = [[pair[1]] for pair in pos_pairs]
        
        
        feed_dict = {
            self.center_words_placeholder: center_words,
            self.target_words_placeholder: target_words,
        }
        
        _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        
        return loss_value



        