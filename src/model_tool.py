from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import os
import pandas as pd
import numpy as np

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn import metrics

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop, Adam

DEFAULT_EMBEDDINGS_PATH = '../data/glove.6B/glove.6B.100d.txt'
DEFAULT_MODEL_DIR = '../models'

MAX_SEQUENCE_LENGTH = 250
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 100
EMBEDDING_TRAINABLE = False
LEARNING_RATE = 0.0001

BATCH_SIZE = 128
EPOCHS = 6
DROPOUT_RATE = 0.3
CNN_FILTER_SIZES = [128, 128]
CNN_KERNEL_SIZES = [5,5]
CNN_POOLING_SIZES = [5, 200] 


def compute_auc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

class ToxModel():
    def __init__(self, 
                 model_name = None, 
                 model_dir = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        if model_name:
            self.load_model_from_name(model_name)
        
    def get_model_name(self):
        return self.model_name

    def load_model_from_name(self, model_name):
        self.model = load_model(os.path.join(self.model_dir, '%s_model.h5' % model_name))
        self.tokenizer = cPickle.load(open(os.path.join(self.model_dir, 
                                                        '%s_tokenizer.pkl' % model_name), 
                                           'rb'))

    def prep_data(self, data_path, text_column, label_column = None, 
                  max_sequence_length = MAX_SEQUENCE_LENGTH, is_training = False):
        """Loads the data from a csv.
        
        Args:
            data_path: path to a csv file of the data to evaluate on.
            text_column: column containing comment text in the csv.
            label_column: column containing toxicity label in the csv. 
                          If set to None, it is assumed the data is unlabelled.
            max_sequence_length: The fixed length of an input sequence. Text will be
                          cropped or padded to meet this length.
            is_training: Indicate if the data is training data on which to initialize
                          the tokenizer.
        
        Returns:
            text_data: A tokenized and padded text sequence as a model input.
            text_labels: A set of labels corresponding to the input text.
                         Returns None if no label_column is set.
        """
        
        data = pd.read_csv(data_path)
        text = data[text_column]
        if is_training:
            self.tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
            self.tokenizer.fit_on_texts(text)
            self.word_index = self.tokenizer.word_index
            cPickle.dump(self.tokenizer, open(os.path.join(self.model_dir, '%s_tokenizer.pkl' % self.model_name), 'wb'))
        text_sequences = self.tokenizer.texts_to_sequences(text)
        text_data = pad_sequences(text_sequences, maxlen = max_sequence_length)
        
        text_labels = None
        if label_column:
            text_labels = to_categorical(data[label_column])
            
        return text_data, text_labels

    def load_embeddings(self, embedding_path = DEFAULT_EMBEDDINGS_PATH):
        self.embeddings_index = {}
        f = open(embedding_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        self.embedding_matrix = np.zeros((len(self.word_index) + 1, EMBEDDING_DIM))
        num_words_in_embedding = 0
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                num_words_in_embedding += 1
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def train(self, training_data_path, validation_data_path, text_column, label_column, model_name):
        self.model_name = model_name

        print('Preparing data...')
        train_data, train_labels = self.prep_data(training_data_path, text_column, label_column, is_training = True)
        valid_data, valid_labels = self.prep_data(validation_data_path, text_column, label_column)
        print('Data prepared!')

        print('Loading embeddings...')
        self.load_embeddings()
        print('Embeddings loaded!')

        print('Building model graph...')
        self.build_model()
        print('Training model...')
        print(self.model.fit(train_data, train_labels,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(valid_data, valid_labels)))
        print('Model trained!')
        print('Saving model...')
        self.model.save(os.path.join(self.model_dir, '%s_model.h5' % self.model_name))
        print('Model saved!')

    def build_model(self):
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(self.word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=EMBEDDING_TRAINABLE)

        embedded_sequences = embedding_layer(sequence_input)
        x = embedded_sequences
        for filter_size, kernel_size, pool_size in zip(CNN_FILTER_SIZES, CNN_KERNEL_SIZES, CNN_POOLING_SIZES):
            x = self.build_conv_layer(x, filter_size, kernel_size, pool_size)
            
        x = Flatten()(x)
        x = Dropout(DROPOUT_RATE)(x)
        # TODO(nthain): Parametrize the number and size of fully connected layers
        x = Dense(128, activation='relu')(x)
        preds = Dense(2, activation='softmax')(x)

        rmsprop = RMSprop(lr = LEARNING_RATE)
        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=rmsprop,
                      metrics=['acc'])        

    def build_conv_layer(self, input_tensor, filter_size, kernel_size, pool_size):
        output = Conv1D(filter_size, kernel_size, activation='relu', padding='same')(input_tensor)
        if pool_size:
            output = MaxPooling1D(pool_size, padding = 'same')(output)
        else:
            # TODO(nthain): This seems broken. Fix.
            output = GlobalMaxPooling1D()(output)
        return output
            
    def predict(self, data):
        return self.model.predict(data)[:,1]
    
    def prep_data_and_predict(self, data_path, text_column, label_column = None):
        data, labels = self.prep_data(data_path, text_column, label_column)
        return self.predict(data)
    
    def prep_data_and_score(self, data_path, text_column, label_column = None):
        data, labels = self.prep_data(data_path, text_column, label_column)
        preds = self.predict(data)
        return compute_auc(labels[:,1], preds)
        
    def summary():
        return self.model.summary()
        
    