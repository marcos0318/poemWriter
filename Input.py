import numpy
import json
from collections import deque

numpy.random.seed(12345)


class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.
    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: number of different words in files, without low-frequency words. 
        sentence_length: The number of words(charactor) in the data file
    """

    def __init__(self, file_name, vocab_size):

        # Constructor
        # First Run get words to 
        self.input_file_name = file_name
        self.get_words(vocab_size)
        # self.init_sample_table()
        self.word_pair_catch = deque()
        print('Vocab size: %d' % len(self.word2id))
        print('Corpus Length: %d' % (self.sentence_length))



    def get_words(self, vocab_size):
        self.input_file = open(self.input_file_name)

        # Lines Counting (Optional)
        num_lines = sum(1 for line in self.input_file)
        print("Total Lines: ", num_lines)

        
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()


        for line in self.input_file:
            
            self.sentence_count += 1

            line = ''.join(line.split()[2:])
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()

        total_frequency = numpy.sort(numpy.array(list(word_frequency.values())))[::-1]
        try:
            min_count = total_frequency[vocab_size]
        except:
            min_count = 0

        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1


        self.word_count = len(self.word2id)

        
    def init_sample_table(self):
        # initialize a sample table for negative sampling. This is much much faster than use np.choice with given probabilities
        # it looks stupid but works really well even using enwiki (around 10G)
        # However need more time when initialization, but worthy
        size = 100000000
        self.sample_table = list()
        sample_dict = {key: value ** 0.75 for key, value in self.word_frequency.items()}
        normalizer = numpy.array(list(sample_dict.values())).sum()
        sample_dict = {key: value / normalizer for key, value in sample_dict.items()}
        for key, value in sample_dict.items():
            number_of_this_word = int(round(value * size))
            self.sample_table.extend([key] * number_of_this_word)
        self.sample_table = numpy.array(self.sample_table)
        self.sample_table_size = self.sample_table.shape[0]
        print("sample table inited with " + str(self.sample_table_size) +" items")

    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()

            # handle like this means the file reach the end. We continue to get the batch from the start
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name)
                sentence = self.input_file.readline()

            word_ids = []
            # The first two sub sentencs is the name and author, so from 2 on counts in
            for word in ''.join(sentence.split()[2:]):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs


    def get_neg_v_neg_sampling(self, pos_word_pair, count):

        """
        :param pos_word_pair: actually only use the number of word pairs
        :param count: the number of negative samples in one
        :return:
        """

        """
        neg_v = numpy.random.choice(
            list(self.sample_table.keys()), size=(len(pos_word_pair), count),
            p=list(self.sample_table.values())).tolist()
        """
        length = len(pos_word_pair)
        indices = numpy.random.randint(self.sample_table_size, size=length*count)
        neg_v = self.sample_table[indices].reshape((length, count))

        return neg_v

    def evaluate_pair_count(self, window_size):
        # total number of pairs in this file = total number of words(as center words) * number of rest words in the window 
        # because not all windows are full sized, need to minus 
        return self.sentence_length * (2 * window_size - 1) - (
                self.sentence_count - 1) * (1 + window_size) * window_size

def test():
    a = InputData('./data/QuanTangShi.txt', 10000)
    return a
