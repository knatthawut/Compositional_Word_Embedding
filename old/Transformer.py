'''
    Implementation the Transformer model
'''
#Import Libraries
import tensorflow as tf
from from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, GRU, Bidirectional, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from gensim.models import Word2Vec
import functools
import numpy as np
import sys
import os
import pprint
from keras.preprocessing.text import Tokenizer
pp = pprint.PrettyPrinter(indent=4)
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from keras_self_attention import SeqSelfAttention
from Keras_baseline import KERAS_baseline
from keras_transformer import get_encoder_component
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,embed_matrix):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,weights=[embed_matrix],input_length=maxlen,trainable=False)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Baseline: the Transformer
class Transformer_baseline(KERAS_baseline):
    '''
    Transformer baseline
    '''
    def __init__(self, attention_activation,  type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH = 21,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam',activation_func = 'tanh',num_heads=1,hidden_dim=100,dropout_rate=0.0):
        # Init model attributes
        super().__init__('Transformer', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        # self.print_information()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate


        inputs = layers.Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        embedding_layer = TokenAndPositionEmbedding(self.MAX_SEQUENCE_LENGTH, self.vocab_size, self.embedding_dim,self.embedding_matrix)
        transformer_block = TransformerBlock(self.embedding_dim, self.num_heads, self.hidden_dim)
        x = embedding_layer(inputs)
        x = transformer_block(x)
        outputs = layers.Dense(embedding_dim, activation=self.activation_func)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)                       
        # Add Dense layer with embedding_dim hidden units to return the vector.

        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use


    def print_information(self):
        super().print_information()
        print(self.model.summary())
