import loadData as ld
import tensorflow as tf
import numpy as np
import datetime
from scipy import spatial
from Modals import OneLayerCNN, BiLSTMmax, BiMultiLSTMmax, OneLayerCNNWithBiLSTMMax, DCNN, BiLSTMmaxWithMultichannel
import config
glove_file_path = config.glove_file_path
numberBatch_file_path = config.numberBatch_file_path
SNLI_file_path = config.SNLI_file_path
MultiNLI_file_path = config.MultiNLI_file_path
modal_save_path = config.modal_save_path

batch_size = ld.batch_size
maxLength = ld.maxLength
iterations = 50000

class Train:
    def __init__(self, 
            batch_size,
            word_vector_embedding_dim, 
            output_embedding_dim, 
            glove_word_dict,
            word_vector_embedding,
            learning_rate=0.0003,
            feature_map_size=150,
            output_graph=False,
            l2_constrain=3.0,
            decay_rate=0.96,
            save_path=None,
            isTrain=True,
            useCorrupt=False,
            model='OCRNN'):
        self.model = model
        self.useCorrupt = useCorrupt
        self.isTrain = isTrain
        self.batch_size = batch_size
        self.output_embedding_dim = output_embedding_dim
        self.word_vector_embedding_dim = word_vector_embedding_dim
        self.word_vector_embedding = word_vector_embedding
        self.test_batch_size = 200
        self.feature_map_size = feature_map_size
        self.l2_constrain = l2_constrain
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.glove_word_dict = glove_word_dict
        self.graph()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(
            max_to_keep=4, keep_checkpoint_every_n_hours=2)
        if not save_path:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, save_path)
            print("Model restored.")
        self.output_graph = output_graph
        if output_graph:
            logdir = "tensorboard/" + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S") + "/"
            self.writer = tf.summary.FileWriter(logdir, self.sess.graph)
    
    def graph(self):

        with tf.variable_scope('trainable'):
            if self.model == 'OCRNN':
                self.s1_input, self.s1_encoder = OneLayerCNNWithBiLSTMMax(
                    's1', self.word_vector_embedding, self.word_vector_embedding_dim, [2, 3, 5], self.output_embedding_dim, feature_map_size=self.feature_map_size)
                self.s2_input, self.s2_encoder = OneLayerCNNWithBiLSTMMax(
                    's2', self.word_vector_embedding, self.word_vector_embedding_dim, [2, 3, 5], self.output_embedding_dim, feature_map_size=self.feature_map_size)
            elif self.model == 'Bi-RNN':
                self.s1_input, self.s1_encoder = BiLSTMmax(
                    's1', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)
                self.s2_input, self.s2_encoder = BiLSTMmax(
                    's2', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)
            elif self.model == 'Bi-RNNwithChannel':
                self.s1_input, self.s1_encoder = BiLSTMmaxWithMultichannel(
                    's1', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)
                self.s2_input, self.s2_encoder = BiLSTMmaxWithMultichannel(
                    's2', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)
            elif self.model == 'MultiBi-RNN':
                self.s1_input, self.s1_encoder = BiMultiLSTMmax(
                    's1', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)
                self.s2_input, self.s2_encoder = BiMultiLSTMmax(
                    's2', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)  
            elif self.model == 'Bi-MultiRNNwithChannel':
                self.s1_input, self.s1_encoder = BiMultiLSTMmaxWithMultichannel(
                    's1', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)
                self.s2_input, self.s2_encoder = BiMultiLSTMmaxWithMultichannel(
                    's2', self.word_vector_embedding, self.word_vector_embedding_dim, self.output_embedding_dim)
                
        
        self.output_embeddings = tf.nn.l2_normalize(self.s1_encoder)

        self.labels = tf.placeholder(shape=[None, 3], name="labels", dtype=tf.int32)

        self.concated = tf.concat([self.s1_encoder, self.s2_encoder], 1)
        self.multiplied = tf.multiply(self.s1_encoder, self.s2_encoder)
        self.abssubtract = tf.abs(tf.subtract(self.s1_encoder, self.s2_encoder))

        self.dense_input = tf.concat([self.concated, self.multiplied, self.abssubtract],1)
        w_initializer, b_initializer = tf.random_normal_initializer(
            0., 0.2), tf.random_normal_initializer(0., 0.1)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.l2_constrain)
        self.fc1 = tf.layers.batch_normalization(tf.layers.dense(self.dense_input, 3,
            kernel_initializer=w_initializer, bias_initializer=b_initializer, kernel_regularizer=l2_regularizer))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels, logits=self.fc1),name="loss")
        self.step = tf.placeholder(tf.int64)
        lr = tf.train.exponential_decay(
            self.lr,
            self.step,
            5000,
            self.decay_rate,
            staircase=True
        )
        Aop = tf.train.AdamOptimizer(learning_rate=lr)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable")
        self.optimizer_all = Aop.minimize(self.loss)
        self.optimizer_freeze = Aop.minimize(self.loss, var_list=train_vars)

        # accuracy
        correct_prediction = tf.equal(tf.argmax(self.fc1, 1), tf.argmax(self.labels, 1))
        
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name="accuracy")
        tf.summary.scalar("training_accuracy", self.accuracy)
        tf.summary.scalar("train_loss", self.loss)
        self.train_summary_op = tf.summary.merge_all()
        tf.summary.scalar("test_accuracy", self.accuracy)
        tf.summary.scalar("test_loss", self.loss)
        self.test_summary_op = tf.summary.merge_all()
        self.param_counter()
        print(tf.trainable_variables())

    def train(self, s1, s2, labels, train_datasize, ts1, ts2, tlabels, ttrain_datasize):
        for i in range(iterations):
            s1arr, s2arr,labelarr = ld.get_batch(s1, s2, labels, train_datasize,corrupt=self.useCorrupt)
            if i % 4 == 0:
                self.sess.run([self.optimizer_all], {self.s1_input: s1arr, self.s2_input: s2arr, self.labels: labelarr,self.step:i})
            else:
                self.sess.run([self.optimizer_freeze], {
                              self.s1_input: s1arr, self.s2_input: s2arr, self.labels: labelarr, self.step: i})
            if (i % 100 == 0):
                loss_value, accuracy_value, summary = self.sess.run(
                    [self.loss, self.accuracy, self.train_summary_op],
                    {self.s1_input: s1arr, self.s2_input: s2arr, self.labels: labelarr})
                if self.output_graph:
                    self.writer.add_summary(summary, i)
                print("--------DEV----------")
                print("Iteration: ", i)
                print("loss", loss_value)
                print("acc", accuracy_value)
                ts1arr, ts2arr, tlabelarr = ld.get_batch(ts1, ts2, tlabels, ttrain_datasize)
                loss_value, accuracy_value, summary = self.sess.run(
                    [self.loss, self.accuracy, self.test_summary_op],
                    {self.s1_input: ts1arr, self.s2_input: ts2arr, self.labels: tlabelarr})
                print("--------TEST----------")
                if self.output_graph:
                    self.writer.add_summary(summary, i)
                print("loss", loss_value)
                print("acc", accuracy_value)
            if i > 0 and i % 10000 == 0:
                self.saver.save(self.sess, modal_save_path +
                                'model.ckpt', global_step=i, write_meta_graph=False)

    def embedding_sentence(self, sentences):
        splited = [sentence.strip().lower().split() for sentence in sentences]
        splited = ld.filter_words(splited)
        sents = []
        for line in splited:
            sent = [self.glove_word_dict[word] if word in self.glove_word_dict else self.glove_word_dict['UNK']
                    for word in line[:maxLength]] + [self.glove_word_dict['UNK']] * (maxLength - len(line))
            sents.append(sent)
        embeddings = self.sess.run(self.output_embeddings, feed_dict={
            self.s1_input: sents})
        return embeddings

    def test(self,s1,s2,labels,datasize):
        s1arr, s2arr, labelarr = ld.get_batch(s1, s2, labels, datasize, datasize)
        loss_value, accuracy_value = self.sess.run(
            [self.loss, self.accuracy],
            {self.s1_input: s1arr, self.s2_input: s2arr, self.labels: labelarr})
        print("----------FINAL-TEST----------")
        print("loss", loss_value)
        print("acc", accuracy_value)

    def save(self, path):
        self.saver.save(self.sess, path)
        print('save success')
    
    def param_counter(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(f"param number:{total_parameters}")

    





if __name__ == '__main__':
    #load embeddings
    embeddings, glove_word_dict, glove_dim_size = ld.load_glove_embeddings(glove_file_path)
    # If you want to load numberBatch word embedding, uncomment this line
    # embeddings, glove_word_dict, glove_dim_size = ld.load_numberbatch_embeddings(numberBatch_file_path)

    print(f"{len(embeddings)} vector of words have been load.")

    #load datas
    s1, s2, labels, train_datasize = ld.load_SNLI_and_MultiNLI_data(
        glove_word_dict, SNLI_file_path, MultiNLI_file_path, 'train', 'train',stop_words = True)
    ts1, ts2, tlabels, ttrain_datasize = ld.load_SNLI_data(glove_word_dict, SNLI_file_path, 'test')

    print(f"{train_datasize} pairs of data have been load.")

    #train model
    model = Train(batch_size=batch_size, word_vector_embedding_dim=glove_dim_size, glove_word_dict=glove_word_dict, output_embedding_dim=512, model="Bi-MultiRNNwithChannel",
                        word_vector_embedding=embeddings, output_graph=False,useCorrupt=True)

    model.train(s1, s2, labels, train_datasize,
                ts1, ts2, tlabels, ttrain_datasize)
    model.test(ts1,ts2,tlabels,ttrain_datasize)

    #small test of model
    x = model.embedding_sentence(["Would you like to have some food ? I am really hungry and thirsty , and need some fish and chips !"])
    y = model.embedding_sentence(["Do you want to have some water ? or maybe beer and sandwich ?"])
    z = model.embedding_sentence(["What is the weather today mom ! I have to go school now ."])
    print("x vs y",1 - spatial.distance.cosine(x, y))
    print("x vs z", 1 - spatial.distance.cosine(x, z))
    print("y vs z", 1 - spatial.distance.cosine(y, z))

    # final save
    model.save(modal_save_path + 'final_model.ckpt')
