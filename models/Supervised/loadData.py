import numpy as np
import os
from random import randint
from spacy.lang.en.stop_words import STOP_WORDS

batch_size = 50
maxLength = 30
np.random.seed(2000)


self_define_stop_words = ['a','an','and','are','as','at','be','by','for','from','has','in','is','have','it','its','of', 'the', 'about',
                        'on','that','to','was','were','will','with','am','been','do','did','does','or','some','somehow','someone']
abbreviations_word_map = ["n't","'s","'ve","'ll","'d"]
need_filted_punctuation = '"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'


def load_SNLI_data(glove_dict,SNL_file_path,Dtype:str,stop_words=False):
    """Load SNLI data and process it into index sequence
    
    Arguments:
        glove_dict {dict} -- loaded glove dict
        SNL_file_path {string} -- SNLI file path
        Dtype {str} -- file type, one of dev, test or train
    
    Keyword Arguments:
        stop_words {bool} -- remove stopwords or not (default: {False})
    
    Returns:
        s1,s2,labels -- dictionary of sent1,sent2 and labels
        length of return size
    """

    assert Dtype in ['dev', 'test', 'train']
    s1 = {}
    s2 = {}
    labels = {}
    with open(f"{SNL_file_path}s1.{Dtype}", 'r') as F:
        s1Lines = [x.lower().split() for x in F.readlines()]
    with open(f"{SNL_file_path}s2.{Dtype}", 'r') as F:
        s2Lines = [x.lower().split() for x in F.readlines()]
    with open(f"{SNL_file_path}labels.{Dtype}", 'r') as F:
        labelsLines = F.readlines()
    if stop_words:
        s1Lines = filter_words(s1Lines)
        s2Lines = filter_words(s2Lines)

    for i, line in enumerate(s1Lines):
        s1[i] = [glove_dict[word.lower()] if word in glove_dict else glove_dict['UNK']
                    for word in line[:maxLength]] + [glove_dict['UNK']] * (maxLength - len(line))
    for i, line in enumerate(s2Lines):
        s2[i] = [glove_dict[word.lower()] if word in glove_dict else glove_dict['UNK']
                    for word in line[:maxLength]] + [glove_dict['UNK']] * (maxLength - len(line))
    
    for i, line in enumerate(labelsLines):
        if line.strip() == 'neutral':
            labels[i] = [1, 0, 0]
        elif line.strip() == 'entailment':
            labels[i] = [0, 1, 0]
        else:
            labels[i] = [0, 0, 1]

    assert len(s1) == len(s2) == len(labels)

    return s1,s2,labels,len(s1)


def load_MultiSNLI_data(glove_dict, SNL_file_path, Dtype: str, stop_words=False):
    """Load MultiSNLI data and process it into index sequence
    
    Arguments:
        glove_dict {dict} -- loaded glove dict
        SNL_file_path {string} -- SNLI file path
        Dtype {str} -- file type, one of dev, test or train
    
    Keyword Arguments:
        stop_words {bool} -- remove stopwords or not (default: {False})
    
    Returns:
        s1,s2,labels -- dictionary of sent1,sent2 and labels
        length of return size
    """
    assert Dtype in ['dev.matched', 'dev.mismatched', 'train']
    s1 = {}
    s2 = {}
    labels = {}
    with open(f"{SNL_file_path}s1.{Dtype}", 'r') as F:
        s1Lines = [x.lower().split() for x in F.readlines()]
    with open(f"{SNL_file_path}s2.{Dtype}", 'r') as F:
        s2Lines = [x.lower().split() for x in F.readlines()]
    with open(f"{SNL_file_path}labels.{Dtype}", 'r') as F:
        labelsLines = F.readlines()
    
    if stop_words:
        s1Lines = filter_words(s1Lines)
        s2Lines = filter_words(s2Lines)

    for i, line in enumerate(s1Lines):
        s1[i] = [glove_dict[word.lower()] if word in glove_dict else glove_dict['UNK']
                 for word in line[:maxLength]] + [glove_dict['UNK']] * (maxLength - len(line))
    for i, line in enumerate(s2Lines):
        s2[i] = [glove_dict[word.lower()] if word in glove_dict else glove_dict['UNK']
                 for word in line[:maxLength]] + [glove_dict['UNK']] * (maxLength - len(line))

    for i, line in enumerate(labelsLines):
        if line.strip() == 'neutral':
            labels[i] = [1, 0, 0]
        elif line.strip() == 'entailment':
            labels[i] = [0, 1, 0]
        else:
            labels[i] = [0, 0, 1]

    assert len(s1) == len(s2) == len(labels)

    return s1, s2, labels, len(s1)

def isin(word):
    """punctuation is in word token or not
    
    Arguments:
        word {str} -- a word
    
    Returns:
        [list] -- a list of boolean
    """

    return [True if char in need_filted_punctuation else False for char in word]

def filter_words(sentences):
    """filter stop words
    
    Arguments:
        sentences {list} -- a list of string sentence
    
    Returns:
        [list] -- a list of string sentence removed their stop words
    """

    new_sentences = []
    for sent in sentences:
        n = [word for word in sent if word.lower() not in self_define_stop_words and all(isin(word)) == False ]
        new_sentences.append(n)
    return new_sentences

        

def load_SNLI_and_MultiNLI_data(glove_dict, SNLI_file_path, MultiNLI_file_path, SNLI_Dtype: str,MultiNLI_Dtype:str,stop_words=False):
    """load SNLI and MultiNLI data in the same time
    
    Arguments:
        glove_dict {dict} -- glove word embedding dict
        SNLI_file_path {str} -- SNLI file path
        MultiNLI_file_path {str} -- MultiNLI file path
        SNLI_Dtype {str} -- SNLI file type, should be one of dev , test and train
        MultiNLI_Dtype {str} -- MultiNLI file type, should be one of 'dev.matched', 'dev.mismatched' and 'train'
    
    Keyword Arguments:
        stop_words {bool} -- remove stop words or not (default: {False})
    
    Returns:
        s1,s2,labels -- dictionary of sent1,sent2 and labels
        length of return size
    """

    assert MultiNLI_Dtype in ['dev.matched', 'dev.mismatched', 'train']
    assert SNLI_Dtype in ['dev', 'test', 'train']
    s1 = {}
    s2 = {}
    labels = {}
    with open(f"{SNLI_file_path}s1.{SNLI_Dtype}", 'r') as F:
        s1Lines = [x.lower().split() for x in F.readlines()]
    with open(f"{SNLI_file_path}s2.{SNLI_Dtype}", 'r') as F:
        s2Lines = [x.lower().split() for x in F.readlines()]
    with open(f"{SNLI_file_path}labels.{SNLI_Dtype}", 'r') as F:
        labelsLines = F.readlines()

    with open(f"{MultiNLI_file_path}s1.{MultiNLI_Dtype}", 'r') as F:
        s1Lines.extend([x.lower().split() for x in F.readlines()])
    with open(f"{MultiNLI_file_path}s2.{MultiNLI_Dtype}", 'r') as F:
        s2Lines.extend([x.lower().split() for x in F.readlines()])
    with open(f"{MultiNLI_file_path}labels.{MultiNLI_Dtype}", 'r') as F:
        labelsLines.extend(F.readlines())
    if stop_words:
        s1Lines = filter_words(s1Lines)
        s2Lines = filter_words(s2Lines)

    for i, line in enumerate(s1Lines):
        s1[i] = [glove_dict[word.lower()] if word in glove_dict else glove_dict['UNK']
                 for word in line[:maxLength]] + [glove_dict['UNK']] * (maxLength - len(line))
    for i, line in enumerate(s2Lines):
        s2[i] = [glove_dict[word.lower()] if word in glove_dict else glove_dict['UNK']
                 for word in line[:maxLength]] + [glove_dict['UNK']] * (maxLength - len(line))

    for i, line in enumerate(labelsLines):
        if line.strip() == 'neutral':
            labels[i] = [1, 0, 0]
        elif line.strip() == 'entailment':
            labels[i] = [0, 1, 0]
        else:
            labels[i] = [0, 0, 1]

    assert len(s1) == len(s2) == len(labels)

    return s1, s2, labels, len(s1)


def load_glove_embeddings(glove_file_path):
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.200d.txt"
    RETURN: embeddings: the array containing word vectors
            glove_word_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
            dim: number of embedding dimension
    """
    glove_word_dict = {}
    with open(f"{glove_file_path}", 'r', encoding="utf-8") as F:
        lines = [x.split() for x in F.readlines()]
        words = len(lines)
        dim = len(lines[0]) - 1
        embeddings = np.zeros(shape=(words + 1, dim), dtype=np.float32)
        glove_word_dict['UNK'] = 0
        embeddings[0] = np.random.randn(dim)
        for i, line in enumerate(lines):
            embeddings[i + 1] = list(map(float, line[1:]))
            glove_word_dict[line[0]] = i + 1

    return embeddings, glove_word_dict, dim

def load_numberbatch_embeddings(embedding_file_path):
    numberbathch_word_dict = {}
    with open(f"{embedding_file_path}", 'r', encoding="utf-8") as F:
        lines = [x.split() for x in F.readlines()][1:]
        words = len(lines)
        dim = len(lines[0]) - 1
        embeddings = np.zeros(shape=(words + 1, dim), dtype=np.float32)
        numberbathch_word_dict['UNK'] = 0
        embeddings[0] = np.random.randn(dim)
        for i, line in enumerate(lines):
            embeddings[i + 1] = list(map(float, line[1:]))
            numberbathch_word_dict[line[0]] = i + 1

    return embeddings, numberbathch_word_dict, dim


def get_batch(s1, s2, labels, train_datasize, batch_size=batch_size,corrupt=False,probability=0.5):
    """get a batch of data with replacement
    
    Arguments:
        s1 {dict} -- sentence1 data
        s2 {dict} -- sentence2 data
        labels {dict} -- label data
        train_datasize {int} -- size of training data size
    
    Keyword Arguments:
        batch_size {int} -- size of batch (default: {batch_size})
        corrupt {bool} -- apply corruption on batch or not (default: {False})
        probability {float} -- probability of corruption (default: {0.5})
    
    Returns:
        arr1, arr2, labelsArr -- batch for training
    """

    labelsArr = []
    arr1 = np.zeros([batch_size, maxLength], dtype=np.int32)
    arr2 = np.zeros([batch_size, maxLength], dtype=np.int32)
    for i in range(batch_size):
        num = randint(0, train_datasize - 1)
        if corrupt:
            arr1[i] = _corruptInput(s1[num],probability)
            arr2[i] = _corruptInput(s2[num],probability)
        else:
            arr1[i] = s1[num]
            arr2[i] = s2[num]
        labelsArr.append(labels[num])
        
    return arr1, arr2, labelsArr

def _corruptInput(array, probability):
    """
    Randomly delete a word form sentence's array.
    """
    a = np.array(array)
    if np.random.random() < probability:
        if any(a):
            noZero = np.flatnonzero(a)
            flipindex = np.random.choice(noZero,replace=False,size=1)[0]
            a = np.concatenate((a[:flipindex], a[flipindex + 1:],[0]))
    return a

