'''
Visualize compound word embeddings using Tensorboard
Usage: 
1. Run: python visualize_compound.py --word compounds.txt --vec vec.np --output output_dir
2. Run: tensorboard --logdir=output_dir
Input:
    - compounds.txt: list of compound words: 
        robot_arm
        bullet_train
    - vec.np: numpy file contains the compound words embeddings.

Ouput:
    - output_dir: path to the tensorboard logs.
'''
# Import libs
import os
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.contrib.tensorboard.plugins import projector

def loadWordVecmodel(vec_file):
    return np.load(vec_file)

def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description='Visualize compound word embeddings')
    parser.add_argument('--word',type=str, metavar='', required=True, help='Path to the compound words list file')
    parser.add_argument('--vec', type=str, metavar='', required=True, help='Path to the numpy vector file')
    parser.add_argument('--output', type=str, metavar='', required=True, help='Path to the output dir')
    args = parser.parse_args()
    vocab_file = args.word
    vec_file = args.vec
    output_dir = args.output

    embedding = loadWordVecmodel(vec_file)

    # setup a TensorFlow session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(X, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embedding})

    # write labels: vocab
    vocab = open(vocab_file,'r',encoding='utf-8').read()
    with open(output_dir+'/metadata.tsv','w',encoding='utf-8') as fout:
        fout.write(vocab)

    # create a TensorFlow summary writer
    summary_writer = tf.summary.FileWriter('log', sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
    projector.visualize_embeddings(summary_writer, config)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join('log', "model.ckpt"))

if __name__ == '__main__':
    main()