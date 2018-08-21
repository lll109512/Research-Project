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
from Prepropcess import readSICKDatawithScore
sys.path.append(
    'models/Supervised')
from train import Train
from train_freeze import Train as Train_f
import loadData as ld
import config
glove_file_path = config.glove_file_path
numberBatch_file_path = config.numberBatch_file_path


text_a, text_b, dev_scores = readSICKDatawithScore()
print(f'{len(text_a)} pair of sts data has been load.')

embeddings, glove_word_dict, glove_dim_size = ld.load_glove_embeddings(
    glove_file_path)
# if you want to use number bath embedding, uncomment this line
# embeddings, glove_word_dict, glove_dim_size = ld.load_numberbatch_embeddings(
#     numberBatch_file_path)
#benchmark
print('======================mean benchmark=========================')
benchmarkEmbedding_a = averageEmbedding(embeddings, glove_word_dict, text_a)
benchmarkEmbedding_b = averageEmbedding(embeddings, glove_word_dict, text_b)



sim_scores = [1 - spatial.distance.cosine(benchmarkEmbedding_a[i], benchmarkEmbedding_b[i])
              for i in range(len(benchmarkEmbedding_a))]

pearson_correlation = scipy.stats.pearsonr(sim_scores, dev_scores)
print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
    pearson_correlation[0], pearson_correlation[1]))


print('======================model result=========================')

# MultiBi-RNNwith_Corrupt_SNLI+MultiNLI_stopwords_removed_stop_words_removed_200 dim mc
model = Train(batch_size=ld.batch_size, word_vector_embedding_dim=glove_dim_size, glove_word_dict=glove_word_dict, output_embedding_dim=512, model="Bi-MultiRNNwithChannel", isTrain=False,
              word_vector_embedding=embeddings, save_path='savedModel/MultiBi-RNNwithCorrupt_SNLI+MultiNLI_stop_words_removed_200dim_mc/20180604-110210/model.ckpt-30000')



benchmarkEmbedding_a = model.embedding_sentence(text_a)
benchmarkEmbedding_b = model.embedding_sentence(text_b)

# sim_scores = np.sum(np.multiply(benchmarkEmbedding_a,
#                                 benchmarkEmbedding_b), axis=1)

sim_scores = [1 - spatial.distance.cosine(benchmarkEmbedding_a[i], benchmarkEmbedding_b[i])
              for i in range(len(benchmarkEmbedding_a))]

pearson_correlation = scipy.stats.pearsonr(sim_scores, dev_scores)
print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
    pearson_correlation[0], pearson_correlation[1]))
