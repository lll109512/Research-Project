import numpy as np
import glob
import os
import json
from string import punctuation
import re

NLU_DATA_PATH = 'data/2017-06-custom-intent-engines/'
SICK_DAT_PATH = 'data/SICK/SICK.txt'
MultiNLI_PATH = 'data/MultiNLI'


def read2017NLUData(home_dir_path=NLU_DATA_PATH):
    returnData = {}
    for path in glob.glob(f"{NLU_DATA_PATH}/*/train_*_full.json"):
        with open(path, 'r', encoding="ISO-8859-1") as F:
            data = json.load(F)
            name = list(data.keys())[0]
            returnData[name] = [''.join([i['text'] for i in sentence['data']]) for sentence in data[name]]          
    return returnData


def readSICKData(file_path=SICK_DAT_PATH):
    returnData = []
    with open(file_path,'r') as F:
        for line in F.readlines()[1:]:
            returnData.extend(line.split('\t')[1:3])
    return preprocessSentences(list(set(returnData)))


def readMultiNLIData(home_dir_path=MultiNLI_PATH):
    with open(f'{MultiNLI_PATH}/s1.train') as F1:
        sent1 = F1.readlines()
    with open(f'{MultiNLI_PATH}/s2.train') as F2:
        sent2 = F2.readlines()
    return sent1+sent2
    


def preprocessSentences(sentences):
    return [' '.join(re.findall(r"'m|'v|'d|'s|[\w]+|[.,!?;]", sent.lower())) for sent in sentences]


def readSICKDatawithScore(file_path=SICK_DAT_PATH):
    sent1 = []
    sent2 = []
    score = []
    with open(file_path, 'r') as F:
        for line in F.readlines()[1:]:
            x = line.split('\t')
            sent1.extend(preprocessSentences([x[1]]))
            sent2.extend(preprocessSentences([x[2]]))
            score.extend([float(x[4])])
    return sent1, sent2, score
