import numpy as np
import sklearn
import Prepropcess
import sys
from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import spatial
# add the model path into system path, make sure program can find 'train' and 'loadData' model
sys.path.append(
    'models/Supervised')
from train import Train, glove_file_path
import loadData as ld


maxLength = ld.maxLength

def averageEmbedding(embeddings, word_dict,sentences):
    splited = [sentence.strip().lower().split() for sentence in sentences]
    sents = []
    for line in splited:
        sent = [word_dict[word] if word in word_dict else word_dict['UNK']
                for word in line[:maxLength]] + [word_dict['UNK']] * (maxLength - len(line))
        sents.append(sent)
    
    returnEmbeddings = []
    for sent in sents:
        vector = np.mean([embeddings[idx] for idx in sent],axis=0)
        returnEmbeddings.append(vector)

    return returnEmbeddings


def seq2averageEmbedding(embeddings, word_dict, sentences):
    splited = [sentence.strip().lower().split() for sentence in sentences]
    sents = []
    for line in splited:
        sent = [word_dict[word] if word in word_dict else word_dict['<UNK>']
                for word in line[:maxLength]] + [word_dict['<UNK>']] * (maxLength - len(line))
        sents.append(sent)

    returnEmbeddings = []
    for sent in sents:
        vector = np.mean([embeddings[idx] for idx in sent], axis=0)
        returnEmbeddings.append(vector)

    return returnEmbeddings
