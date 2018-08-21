import numpy as np
import sklearn
import Prepropcess
import pandas
import tensorflow as tf
import sys
import os
from scipy import spatial
import scipy
sys.path.append(
    'models/Supervised')
from train import Train
import loadData as ld
#set glove_file_path
glove_file_path = "data/glove.6B/glove.6B.200d.txt"

embeddings, glove_word_dict, glove_dim_size = ld.load_glove_embeddings(glove_file_path)
print('Loading model...')
# MultiBi-RNNwith_Corrupt_SNLI+MultiNLI_stopwords_removed_stop_words_removed_200 dim mc
model = Train(batch_size=ld.batch_size, word_vector_embedding_dim=glove_dim_size, glove_word_dict=glove_word_dict, output_embedding_dim=512, model="Bi-MultiRNNwithChannel", isTrain=False,
              word_vector_embedding=embeddings, save_path='savedModel/MultiBi-RNNwithCorrupt_SNLI+MultiNLI_stop_words_removed_200dim_mc/20180604-110210/model.ckpt-30000')

sentA = 'A man is writing a fantasy novel'

sentBs = ['A boy is writing a fantasy novel',
          'A man is writing a fantasy fiction',
          'A man has written a fantasy novel',
          'A man is writing a romantic novel',
          'A man is writing a sad novel',
          'A woman is writing a fantasy novel',
          'A man has written several novel',
          'A man is playing the football',
          'A man is not writing a fantasy novel',
          'A man is not writing a not fantasy novel']

# sentC = 'If you like this book , i can buy one for you'

# sentDs = ['I can buy one for you , if you like this book',
#           'I can buy this book for you , if you like one',
#           'If you do not like this book, i can not buy one for you',
#           'I can not buy this book for you , if you do not like this one']

sentC = 'I love you .'

sentDs = ["I like you .",
          "I do n't love you .",
          "I hide you in my heart .",
          'I think I have a crush on you for a long time .',
          'Meeting you was fate , and falling in love with you was out of my control .',
          'The reason why I live so far is to meet you at the moment .',
          'Even if the whole world betrayed you , I will stand beside you betraying the world .',
          'I would like to use ten million years to wait for your early spring warm smile .']


embeddingsA = model.embedding_sentence([sentA])[0]
embeddingsBs = model.embedding_sentence(sentBs)

print(f"Sentence A:{sentA}")
print(f'Sentence B and similarity with A')
for index, embedding in enumerate(embeddingsBs):
    print(f'{sentBs[index]}:',1 - spatial.distance.cosine(embeddingsA, embedding))


embeddingsC = model.embedding_sentence([sentC])[0]
embeddingsDs = model.embedding_sentence(sentDs)

print(f"Sentence C:{sentC}")
print(f'Sentence D and similarity with C')
for index, embedding in enumerate(embeddingsDs):
    print(f'{sentDs[index]}:', 1 - spatial.distance.cosine(embeddingsC, embedding))
