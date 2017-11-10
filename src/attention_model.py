from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Dropout, merge
from keras.models import Model
from keras.optimizers import RMSprop

from model_tool import ToxModel, DEFAULT_MODEL_DIR


class AttentionToxModel(ToxModel):

    def build_dense_attention_layer(self, input_tensor):
        print(self.hparams['max_sequence_length'])
        attention_probs = Dense(self.hparams['max_sequence_length'], activation='softmax', name='attention_vec')(input_tensor)
        attention_mul = merge([input_tensor, attention_probs], output_shape=32, name='attention_mul', mode='mul')
        return attention_mul

    def build_model(self):
        print('print inside build model')
        sequence_input = Input(shape=(self.hparams['max_sequence_length'],), dtype='int32')
        embedding_layer = Embedding(len(self.tokenizer.word_index) + 1,
                                    self.hparams['embedding_dim'],
                                    weights=[self.embedding_matrix],
                                    input_length=self.hparams['max_sequence_length'],
                                    trainable=self.hparams['embedding_trainable'])

        embedded_sequences = embedding_layer(sequence_input)
        x = embedded_sequences
        for filter_size, kernel_size, pool_size in zip(self.hparams['cnn_filter_sizes'], self.hparams['cnn_kernel_sizes'], self.hparams['cnn_pooling_sizes']):
            x = self.build_conv_layer(x, filter_size, kernel_size, pool_size)

        x = Flatten()(x)
        x = Dropout(self.hparams['dropout_rate'])(x)
        # TODO(nthain): Parametrize the number and size of fully connected layers
        x = Dense(250, activation='relu')(x)
        preds = self.build_dense_attention_layer(x)

        rmsprop = RMSprop(lr=self.hparams['learning_rate'])
        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop,metrics=['acc'])
