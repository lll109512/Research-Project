### COMP9596 project usage

##### 1. project structure

There are four fold in this project 

| Fold       | Description                                        |
| ---------- | -------------------------------------------------- |
| data       | The data used in training and evaluation           |
| model      | Tensorflow program for surpervise and unsurpervise |
| Tools      | Some tools for evaluation and visualization        |
| savedModel | The model tained by alogrithm.                     |

For data fold, there will be six fold for different data

| Fold                          | Description                             |
| ----------------------------- | --------------------------------------- |
| 2017-06-custom-intent-engines | NLU data for data cluster visualization |
| glove.6B                      | GloVe pretrain word embedding           |
| MultiNLI                      | MultiNLI data                           |
| SICK                          | SICK data                               |
| SNLI                          | Stanford NLI data                       |
| stsbenchmark                  | STS data                                |

For model fold, I only submit the surpervised model.

| File            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| config.py       | The config file for path setting. There are five required data path in this file. You can set it in this file. |
| loadData.py     | Help function for data preprocessing and sampling.           |
| Models.py       | All encoder for training.                                    |
| train.py        | Training program                                             |
| train_freeze.py | same training program witch added a condiction for training full connection layer every k iteration. |

For tools fold, there are some tools for evaluation.

| File               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| analysis.py        | A tool load sts data to calculate top similarity between them. It will output top 10 similar sentences and the top 10 unsimilar sentence with their similarity |
| benchmark.py       | A function calculate average sentence embedding as a benchmark |
| Preprocess.py      | A file provide preprocessing function for NLU, SICK and MultiNLI data for tool functions. There are three path should be set correct in the begaining of this file. |
| sick_benchmark.py  | A program calculate pearson r by sick data.                  |
| sts_benchmark.py   | A program calculate pearson r by sts data.                   |
| simSentenceTest.py | A program calculate similarity by given sentences with sligtly change. |
| visualize.py       | A program give a cluster plot using NLU data by T-SNE.       |
|                    |                                                              |

##### 2. Requirements

###### 1. python version

python3.6

###### 2. packages

```
numpy
scipy
sklearn
tensorflow 1.6+
scipy 2.0 
```

You can simply install package by requirements.txt

```
pip3 install -r requirements.txt
```

After install spacy, you still need install en language support to run it.

```
python3 -m spacy download en
```
##### 3. Train and run

###### 1. Train

Simply run

```
python3 train.py
or
python3 train_freeze.py
```

The submitted train.py defaultly run the best model I used for training Bi-MultiRNNwithChannel. You can change the **iteration** and **output_embedding_dim** for a faster training.

###### 2. Get embedding

You can use your saved trained model or given model by default to embedding sentences.

```python
from train import Train
import loadData as ld

# 1.load pretrain word embedding
embeddings, glove_word_dict, glove_dim_size = ld.load_glove_embeddings(glove_file_path)

# 2.load model
model = Train(batch_size=ld.batch_size, word_vector_embedding_dim=glove_dim_size, glove_word_dict=glove_word_dict, output_embedding_dim=512, model="Bi-MultiRNNwithChannel", isTrain=False,
              word_vector_embedding=embeddings, save_path='savedModel/MultiBi-RNNwithCorrupt_SNLI+MultiNLI_stop_words_removed_200dim_mc/20180604-110210/model.ckpt-30000')

# 3.embedding sentences
sents = ['sentence A','sentence B']
embeddings = model.embedding_sentence(sents)
```

As a example, you can view the simSentneceTest.py and learn how to use it.

###### 3. Tools

For all tools program, simply run it by

```
python3 analysis.py
or
python3 sick_benchmark.py
or
python3 sts_benchmark.py
or
python3 simSentenceTest.py
ot
python3 visualize.py
```

All program will print result in terminal.

However, for visualize.py, unless you close the first benchmark plot, the second plot will not be rendered.

Be careful, the config.py file for model will also be used in these tools, such as Glove file path. If you want to set different file, you can change it locally. But you should make sure the pretrained embedding and output dimenssion should be the same in your training phase。

