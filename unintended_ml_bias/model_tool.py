"""Train a Toxicity model using Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle
import json
import os
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.models import load_model
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn import metrics

print('HELLO from model_tool')

DEFAULT_EMBEDDINGS_PATH = '../data/glove.6B/glove.6B.100d.txt'
DEFAULT_MODEL_DIR = '../models'

DEFAULT_HPARAMS = {
    'max_sequence_length': 250,
    'max_num_words': 10000,
    'embedding_dim': 100,
    'embedding_trainable': False,
    'learning_rate': 0.00005,
    'stop_early': True,
    'es_patience': 1,  # Only relevant if STOP_EARLY = True
    'es_min_delta': 0,  # Only relevant if STOP_EARLY = True
    'batch_size': 128,
    'epochs': 20,
    'dropout_rate': 0.3,
    'cnn_filter_sizes': [128, 128, 128],
    'cnn_kernel_sizes': [5, 5, 5],
    'cnn_pooling_sizes': [5, 5, 40],
    'verbose': True
}


def compute_auc(y_true, y_pred):
  try:
    return metrics.roc_auc_score(y_true, y_pred)
  except ValueError:
    return np.nan


### Model scoring

# Scoring these dataset for dozens of models actually takes non-trivial amounts
# of time, so we save the results as a CSV. The resulting CSV includes all the
# columns of the original dataset, and in addition has columns for each model,
# containing the model's scores.
def score_dataset(df, models, text_col):
    """Scores the dataset with each model and adds the scores as new columns."""
    for model in models:
        name = model.get_model_name()
        print('{} Scoring with {}...'.format(datetime.datetime.now(), name))
        df[name] = model.predict(df[text_col])

def load_maybe_score(models, orig_path, scored_path, postprocess_fn):
    if os.path.exists(scored_path):
        print('Using previously scored data:', scored_path)
        return pd.read_csv(scored_path)

    dataset = pd.read_csv(orig_path)
    postprocess_fn(dataset)
    score_dataset(dataset, models, 'text')
    print('Saving scores to:', scored_path)
    dataset.to_csv(scored_path)
    return dataset

def postprocess_madlibs(madlibs):
    """Modifies madlibs data to have standard 'text' and 'label' columns."""
    # Native madlibs data uses 'Label' column with values 'BAD' and 'NOT_BAD'.
    # Replace with a bool.
    madlibs['label'] = madlibs['Label'] == 'BAD'
    madlibs.drop('Label', axis=1, inplace=True)
    madlibs.rename(columns={'Text': 'text'}, inplace=True)

def postprocess_wiki_dataset(wiki_data):
    """Modifies Wikipedia dataset to have 'text' and 'label' columns."""
    wiki_data.rename(columns={'is_toxic': 'label',
                              'comment': 'text'},
                     inplace=True)


class ToxModel():
  """Toxicity model."""

  def __init__(self,
               model_name=None,
               model_dir=DEFAULT_MODEL_DIR,
               embeddings_path=DEFAULT_EMBEDDINGS_PATH,
               hparams=None):
    self.model_dir = model_dir
    self.embeddings_path = embeddings_path
    self.model_name = model_name
    self.model = None
    self.tokenizer = None
    self.hparams = DEFAULT_HPARAMS.copy()
    if hparams:
      self.update_hparams(hparams)
    if model_name:
      self.load_model_from_name(model_name)
    self.print_hparams()

  def print_hparams(self):
    print('Hyperparameters')
    print('---------------')
    for k, v in self.hparams.iteritems():
      print('{}: {}'.format(k, v))
    print('')

  def update_hparams(self, new_hparams):
    self.hparams.update(new_hparams)

  def get_model_name(self):
    return self.model_name

  def save_hparams(self, model_name):
    self.hparams['model_name'] = model_name
    with open(
        os.path.join(self.model_dir, '%s_hparams.json' % self.model_name),
        'w') as f:
      json.dump(self.hparams, f, sort_keys=True)

  def load_model_from_name(self, model_name):
    self.model = load_model(
        os.path.join(self.model_dir, '%s_model.h5' % model_name))
    self.tokenizer = cPickle.load(
        open(
            os.path.join(self.model_dir, '%s_tokenizer.pkl' % model_name),
            'rb'))
    with open(
        os.path.join(self.model_dir, '%s_hparams.json' % self.model_name),
        'r') as f:
      self.hparams = json.load(f)

  def fit_and_save_tokenizer(self, texts):
    """Fits tokenizer on texts and pickles the tokenizer state."""
    self.tokenizer = Tokenizer(num_words=self.hparams['max_num_words'])
    self.tokenizer.fit_on_texts(texts)
    cPickle.dump(self.tokenizer,
                 open(
                     os.path.join(self.model_dir,
                                  '%s_tokenizer.pkl' % self.model_name), 'wb'))

  def prep_text(self, texts):
    """Turns text into into padded sequences.

    The tokenizer must be initialized before calling this method.

    Args:
      texts: Sequence of text strings.

    Returns:
      A tokenized and padded text sequence as a model input.
    """
    text_sequences = self.tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        text_sequences, maxlen=self.hparams['max_sequence_length'])

  def load_embeddings(self):
    """Loads word embeddings."""
    embeddings_index = {}
    with open(self.embeddings_path) as f:
      for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    self.embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1,
                                      self.hparams['embedding_dim']))
    num_words_in_embedding = 0
    for word, i in self.tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        num_words_in_embedding += 1
        # words not found in embedding index will be all-zeros.
        self.embedding_matrix[i] = embedding_vector

  def train(self, training_data_path, validation_data_path, text_column,
            label_column, model_name):
    """Trains the model."""
    self.model_name = model_name
    self.save_hparams(model_name)

    train_data = pd.read_csv(training_data_path)
    valid_data = pd.read_csv(validation_data_path)

    print('Fitting tokenizer...')
    self.fit_and_save_tokenizer(train_data[text_column])
    print('Tokenizer fitted!')

    print('Preparing data...')
    train_text, train_labels = (self.prep_text(train_data[text_column]),
                                to_categorical(train_data[label_column]))
    valid_text, valid_labels = (self.prep_text(valid_data[text_column]),
                                to_categorical(valid_data[label_column]))
    print('Data prepared!')

    print('Loading embeddings...')
    self.load_embeddings()
    print('Embeddings loaded!')

    print('Building model graph...')
    self.build_model()
    print('Training model...')

    save_path = os.path.join(self.model_dir, '%s_model.h5' % self.model_name)
    callbacks = [
        ModelCheckpoint(
            save_path, save_best_only=True, verbose=self.hparams['verbose'])
    ]

    if self.hparams['stop_early']:
      callbacks.append(
          EarlyStopping(
              min_delta=self.hparams['es_min_delta'],
              monitor='val_loss',
              patience=self.hparams['es_patience'],
              verbose=self.hparams['verbose'],
              mode='auto'))

    self.model.fit(
        train_text,
        train_labels,
        batch_size=self.hparams['batch_size'],
        epochs=self.hparams['epochs'],
        validation_data=(valid_text, valid_labels),
        callbacks=callbacks,
        verbose=2)
    print('Model trained!')
    print('Best model saved to {}'.format(save_path))
    print('Loading best model from checkpoint...')
    self.model = load_model(save_path)
    print('Model loaded!')

  def build_model(self):
    """Builds model graph."""
    sequence_input = Input(
        shape=(self.hparams['max_sequence_length'],), dtype='int32')
    embedding_layer = Embedding(
        len(self.tokenizer.word_index) + 1,
        self.hparams['embedding_dim'],
        weights=[self.embedding_matrix],
        input_length=self.hparams['max_sequence_length'],
        trainable=self.hparams['embedding_trainable'])

    embedded_sequences = embedding_layer(sequence_input)
    x = embedded_sequences
    for filter_size, kernel_size, pool_size in zip(
        self.hparams['cnn_filter_sizes'], self.hparams['cnn_kernel_sizes'],
        self.hparams['cnn_pooling_sizes']):
      x = self.build_conv_layer(x, filter_size, kernel_size, pool_size)

    x = Flatten()(x)
    x = Dropout(self.hparams['dropout_rate'])(x)
    # TODO(nthain): Parametrize the number and size of fully connected layers
    x = Dense(128, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)

    rmsprop = RMSprop(lr=self.hparams['learning_rate'])
    self.model = Model(sequence_input, preds)
    self.model.compile(
        loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])

  def build_conv_layer(self, input_tensor, filter_size, kernel_size, pool_size):
    output = Conv1D(
        filter_size, kernel_size, activation='relu', padding='same')(
            input_tensor)
    if pool_size:
      output = MaxPooling1D(pool_size, padding='same')(output)
    else:
      # TODO(nthain): This seems broken. Fix.
      output = GlobalMaxPooling1D()(output)
    return output

  def predict(self, texts):
    """Returns model predictions on texts."""
    data = self.prep_text(texts)
    return self.model.predict(data)[:, 1]

  def score_auc(self, texts, labels):
    preds = self.predict(texts)
    return compute_auc(labels, preds)

  def summary(self):
    return self.model.summary()
