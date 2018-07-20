import tensorflow as tf
import numpy as np
import pandas as pd
from Input import InputData 
from tqdm import tqdm
from model import Word2Vec

embedding_size = 300
data_path = './data/QuanTangShi.txt'


word2vec = Word2Vec(data_path, embedding_size)

iteration = 0.01
pair_count = word2vec.data.evaluate_pair_count(word2vec.window_size)
batch_count = iteration * pair_count / word2vec.batch_size
process_bar = tqdm(range(int(batch_count)))

losses = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in process_bar:
        loss = word2vec.train_epoch(sess)
        process_bar.set_description("Loss: %0.8f" %loss)
        losses.append(loss)
        
    final_embedding = word2vec.normalized_embeddings.eval()
    print("training finished")


labels = [word2vec.data.id2word[i] for i in range(len(word2vec.data.id2word))]
df = pd.DataFrame(data=final_embedding, index=labels, columns=list(range(1,301))) 

df.to_csv("embedding.csv")
print("saving finished")