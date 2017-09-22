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

def compute_auc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

class Model():
    def __init__(self, 
                 model_name, 
                 model_dir = '../models'):
        self.model_dir = model_dir
        self.model = load_model(os.path.join(self.model_dir, '%s_model.h5' % model_name))
        self.tokenizer = cPickle.load(open(os.path.join(self.model_dir, 
                                                        '%s_tokenizer.pkl' % model_name), 
                                           'rb'))
        
    def prep_data(self, data_path, text_column, label_column = None, max_sequence_length = 1000):
        """Loads the data from a csv.
        
        Args:
            data_path: path to a csv file of the data to evaluate on.
            text_column: column containing comment text in the csv.
            label_column: column containing toxicity label in the csv. 
                          If set to None, it is assumed the data is unlabelled.
        
        Returns:
            text_data: A tokenized and padded text sequence as a model input.
            text_labels: A set of labels corresponding to the input text.
                         Returns None if no label_column is set.
        """
        
        data = pd.read_csv(data_path)
        text = data[text_column]
        text_sequences = self.tokenizer.texts_to_sequences(text)
        text_data = pad_sequences(text_sequences, maxlen = max_sequence_length)
        
        text_labels = None
        if label_column:
            text_labels = to_categorical(data[label_column])[:,1]
            
        return text_data, text_labels
            
    def predict(self, data):
        return self.model.predict(data)[:,1]
    
    def prep_data_and_predict(self, data_path, text_column, label_column = None):
        data, labels = self.prep_data(data_path, text_column, label_column)
        return self.predict(data)
    
    def prep_data_and_score(self, data_path, text_column, label_column = None):
        data, labels = self.prep_data(data_path, text_column, label_column)
        preds = self.predict(data)
        return compute_auc(labels, preds)
        
        
    