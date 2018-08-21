import numpy as np
import sklearn
import Prepropcess
import pandas
import tensorflow as tf
import sys
import os
from scipy import spatial
import scipy
from benchmark import averageEmbedding
sys.path.append(
    'models/Supervised')
from train import Train
from train_freeze import Train as Train_f
import loadData as ld
import config
glove_file_path = config.glove_file_path
sts_dataset = 'data/'
numberBatch_file_path = config.numberBatch_file_path

def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
        # (sent_1, sent_2, similarity_score)
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dev = load_sts_dataset(
        os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(
        os.path.join(
            os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))
    sts_train = load_sts_dataset(
        os.path.join(
            os.path.dirname(sts_dataset), "stsbenchmark", "sts-train.csv"))

    return sts_dev, sts_test, sts_train



sts_dev, sts_test, sts_train = download_and_load_sts_data()
text_a = sts_dev['sent_1'].tolist()
text_b = sts_dev['sent_2'].tolist()
dev_scores = sts_dev['sim'].tolist()
print(f'{len(text_a)} pair of sts data has been load.')

embeddings, glove_word_dict, glove_dim_size = ld.load_glove_embeddings(
    glove_file_path)
# embeddings, glove_word_dict, glove_dim_size = ld.load_numberbatch_embeddings(
#     numberBatch_file_path)
#benchmark
print('======================mean benchmark=========================')
benchmarkEmbedding_a = averageEmbedding(embeddings, glove_word_dict, text_a)
benchmarkEmbedding_b = averageEmbedding(embeddings, glove_word_dict, text_b)

    
sim_scores = [1 - spatial.distance.cosine(benchmarkEmbedding_a[i], benchmarkEmbedding_b[i]) for i in range(len(benchmarkEmbedding_a))]

pearson_correlation = scipy.stats.pearsonr(sim_scores, dev_scores)
print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
    pearson_correlation[0], pearson_correlation[1]))


print('======================model result=========================')

# MultiBi-RNNwith_Corrupt_SNLI+MultiNLI_stopwords_removed_stop_words_removed_200 dim mc
model = Train(batch_size=ld.batch_size, word_vector_embedding_dim=glove_dim_size, glove_word_dict=glove_word_dict, output_embedding_dim=512, model="Bi-MultiRNNwithChannel", isTrain=False,
              word_vector_embedding=embeddings, save_path='savedModel/MultiBi-RNNwithCorrupt_SNLI+MultiNLI_stop_words_removed_200dim_mc/20180604-110210/model.ckpt-30000')
benchmarkEmbedding_a = model.embedding_sentence(text_a)
benchmarkEmbedding_b = model.embedding_sentence(text_b)



sim_scores = [1 - spatial.distance.cosine(benchmarkEmbedding_a[i], benchmarkEmbedding_b[i])
              for i in range(len(benchmarkEmbedding_a))]

pearson_correlation = scipy.stats.pearsonr(sim_scores, dev_scores)
print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
    pearson_correlation[0], pearson_correlation[1]))
