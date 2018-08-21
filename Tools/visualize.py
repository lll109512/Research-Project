import numpy as np
import sklearn
import Prepropcess
import sys
from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
sys.path.append('models/Supervised')
from train import Train
import loadData as ld
from benchmark import averageEmbedding
import config

SAMPLE_NUMBER = 50
glove_file_path = config.glove_file_path

#configue
Marker = {'PlayMusic': 'o', 'RateBook': 'v', 'SearchCreativeWork': '^', 'GetWeather': '8',
          'BookRestaurant': 's', 'AddToPlaylist': 'p', 'SearchScreeningEvent': '+'}
Color = {'PlayMusic': 'r', 'RateBook': 'chocolate', 'SearchCreativeWork': 'orange', 'GetWeather': 'forestgreen',
         'BookRestaurant': 'dodgerblue', 'AddToPlaylist': 'darkviolet', 'SearchScreeningEvent': 'deeppink'}

embeddings, glove_word_dict, glove_dim_size = ld.load_glove_embeddings(glove_file_path)


NLUdata = Prepropcess.read2017NLUData()
selectedData = {}
for key, value in NLUdata.items():
    if key in ['BookRestaurant', 'GetWeather', 'PlayMusic']:
        selectedData[key] = np.random.choice(np.array(value), SAMPLE_NUMBER,replace=False)

#benchmark embedding
print("===============Mean Benchmark===============")
benchmarkEmbeddings = {}
for key, value in selectedData.items():
    benchmarkEmbeddings[key] = averageEmbedding(
        embeddings, glove_word_dict, value)


#visualize
datas = []
keys = []
for key in benchmarkEmbeddings:
    datas.append(benchmarkEmbeddings[key])
    keys.append(key)

concatenated = np.concatenate(datas, axis=0)


prepca = decomposition.PCA(n_components=50, copy=True)
decomposedaAllAata = prepca.fit_transform(concatenated)
preDecomposedData = {}
for key, value in benchmarkEmbeddings.items():
    preDecomposedData[key] = prepca.transform(value)

tsne = TSNE(n_components=2, perplexity=20, learning_rate=50, n_iter=3000)
tsneResult = tsne.fit_transform(decomposedaAllAata)

for i, key in enumerate(keys):
    plt.scatter(tsneResult[i * SAMPLE_NUMBER:(i + 1) * SAMPLE_NUMBER, 0],
                tsneResult[i * SAMPLE_NUMBER:(i + 1) * SAMPLE_NUMBER, 1], marker=Marker[key], label=key, c=Color[key])
    for k, v in enumerate(selectedData[key]):
        plt.text(tsneResult[i * SAMPLE_NUMBER + k, 0],
                 tsneResult[i * SAMPLE_NUMBER + k, 1], v, fontdict={'size': 6, 'color': Color[key]})
plt.legend(loc='upper right')
plt.show()

print("===============Model Result===============")

model = Train(batch_size=ld.batch_size, word_vector_embedding_dim=glove_dim_size, glove_word_dict=glove_word_dict, output_embedding_dim=512, model="Bi-MultiRNNwithChannel", isTrain=False,
              word_vector_embedding=embeddings, save_path='savedModel/MultiBi-RNNwithCorrupt_SNLI+MultiNLI_stop_words_removed_200dim_mc/20180604-110210/model.ckpt-30000')

dataEmbeddings = {}
for key ,value in selectedData.items():
    dataEmbeddings[key] = model.embedding_sentence(value)

#visualize
datas = []
keys = []
for key in dataEmbeddings:
    datas.append(dataEmbeddings[key])
    keys.append(key)

concatenated = np.concatenate(datas, axis=0)

#PCA
# pca = decomposition.PCA(n_components=2,copy=True)
# pca.fit(concatenated)
# decomposedData = {}
# for key, value in dataEmbeddings.items():
#     decomposedData[key] = pca.transform(value)

# # print(decomposedData)
# for key, value in decomposedData.items():
#     plt.scatter(value[:, 0], value[:, 1],
#                 marker=Marker[key], label=key, c=Color[key])
#     for i, v in enumerate(selectedData[key]):
#         plt.text(value[i, 0], value[i, 1], v,
#                  fontdict={'size': 6, 'color': Color[key]})
# plt.legend(loc='upper right')
# plt.savefig('data.svg', format='svg')
# plt.show()


#T-SNE
prepca = decomposition.PCA(n_components=50,copy=True)
decomposedaAllAata = prepca.fit_transform(concatenated)
preDecomposedData = {}
for key, value in dataEmbeddings.items():
    preDecomposedData[key] = prepca.transform(value)

tsne = TSNE(n_components=2, perplexity=20, learning_rate=50, n_iter=3000)
tsneResult = tsne.fit_transform(decomposedaAllAata)

for i, key in enumerate(keys):
    plt.scatter(tsneResult[i * SAMPLE_NUMBER:(i + 1) * SAMPLE_NUMBER, 0],
                tsneResult[i * SAMPLE_NUMBER:(i + 1) * SAMPLE_NUMBER, 1], marker=Marker[key],label=key,c=Color[key])
    #If you want to see the sentence in graph, uncomment following line

    # for k,v in enumerate(selectedData[key]):
    #     plt.text(tsneResult[i * SAMPLE_NUMBER + k, 0],
    #              tsneResult[i * SAMPLE_NUMBER + k, 1], v, fontdict={'size': 6, 'color': Color[key]})
plt.legend(loc='upper right')
plt.show()
