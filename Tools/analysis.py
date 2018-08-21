import numpy as np
import sklearn
import Prepropcess
import sys
from scipy import spatial
import scipy
from benchmark import averageEmbedding
# add the model path into system path, make sure program can find 'train' and 'loadData' model
sys.path.append(
    'models/Supervised')
from train import Train
from train_freeze import Train as Train_f
import loadData as ld
import config
glove_file_path = config.glove_file_path
numberBatch_file_path = config.numberBatch_file_path

# the number of top sentence with heighest similarity
top_k = 10

target_sentances = ['a woman is slicing potatoes',
                    'a boy is waving at some young runners from the ocean',
                    'two men are playing guitar']

#load SICK data.
SICKData = Prepropcess.readSICKData()
print(f"{len(SICKData)} has been load.")

#load word embedding
embeddings, glove_word_dict, glove_dim_size = ld.load_glove_embeddings(
    glove_file_path)
# if you want to use number bath embedding, uncomment this line
# embeddings, glove_word_dict, glove_dim_size = ld.load_numberbatch_embeddings(
#     numberBatch_file_path)

#Benchmark
print("============Mean benchmark result============")

benchmarkEmbedding =  averageEmbedding(embeddings, glove_word_dict, SICKData)
print(f"{len(benchmarkEmbedding)} vector have been computed")
benchmarkTargetEmbedding = averageEmbedding(embeddings, glove_word_dict, target_sentances)
print(f"{len(benchmarkTargetEmbedding)} benchmark sentence have been load.")

BenchmarkSimilarity = []
for benchmark in benchmarkTargetEmbedding:
    temp_sim = []
    for idx, embedding in enumerate(benchmarkEmbedding):
        temp_sim.append(
            [1 - spatial.distance.cosine(benchmark, embedding), SICKData[idx]])
    BenchmarkSimilarity.append(temp_sim)


#select top k
print(f"Top {top_k} similar sentence:")
for idx, benchmark in enumerate(target_sentances):
    print(f'Sentence:  {benchmark}')
    topkS = sorted(BenchmarkSimilarity[idx], key=lambda x: x[0], reverse=True)[1:top_k + 1]
    for sent in topkS:
        print(f"Similarity: {sent[0]:2f}    Sentence: {sent[1]}")
    print("----------------")

    x = sorted(BenchmarkSimilarity[idx], key=lambda x: x[0])[1:top_k + 1]
    print(x)

print("=================Model result================")

# MultiBi-RNNwith_Corrupt_SNLI+MultiNLI_stopwords_removed_stop_words_removed_200 dim mc
model = Train(batch_size=ld.batch_size, word_vector_embedding_dim=glove_dim_size, glove_word_dict=glove_word_dict, output_embedding_dim=512, model="Bi-MultiRNNwithChannel", isTrain=False,
              word_vector_embedding=embeddings, save_path='savedModel/MultiBi-RNNwithCorrupt_SNLI+MultiNLI_stop_words_removed_200dim_mc/20180604-110210/model.ckpt-30000')
              
dataEmbeddings = []
dataEmbeddings = model.embedding_sentence(SICKData)
print(f"{len(dataEmbeddings)} vector have been computed")

targetSentsEmbeddings = model.embedding_sentence(target_sentances)
print(f"{len(target_sentances)} benchmark sentence have been load.")

similarity = []
for benchmark in targetSentsEmbeddings:
    temp_sim = []
    for idx, embedding in enumerate(dataEmbeddings):
        temp_sim.append(
            [1 - spatial.distance.cosine(benchmark, embedding), SICKData[idx]])
    similarity.append(temp_sim)


#select top k
print(f"Top {top_k} similar sentence:")
for idx, benchmark in enumerate(target_sentances):
    print(f'Sentence:  {benchmark}')
    topkS = sorted(similarity[idx],key=lambda x:x[0],reverse=True)[1:top_k+1]
    for sent in topkS:
        print(f"Similarity: {sent[0]:2f}    Sentence: {sent[1]}")
    print("----------------")

    x = sorted(similarity[idx], key=lambda x: x[0])[1:top_k + 1]
    print(x)

