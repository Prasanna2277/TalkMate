import tensorflow as tf
import numpy as np
import re
from collections import Counter
import sys
import math
from random import randint
import pickle
import os

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
NEGATIVE_SAMPLES = 64
CONTEXT_WINDOW = 5
NUM_TRAIN_STEPS = 100000

# Process dataset into word frequency dictionary
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    text_corpus = "".join(lines)
    word_freq = Counter(text_corpus.split())
    return text_corpus, word_freq

# Create skip-gram training pairs
def build_training_data(word_freq, corpus_text):
    vocab = list(word_freq.keys())
    tokens = corpus_text.split()
    total_tokens = len(tokens)
    input_words, target_words = [], []

    for idx in range(total_tokens):
        if idx % 100000 == 0:
            print("Processed %d/%d tokens" % (idx, total_tokens))

        context_after = tokens[idx + 1: idx + CONTEXT_WINDOW + 1]
        context_before = tokens[max(0, idx - CONTEXT_WINDOW): idx]
        context_words = context_after + context_before

        for context_word in context_words:
            input_words.append(vocab.index(tokens[idx]))
            target_words.append(vocab.index(context_word))

    return input_words, target_words

# Batch generator
def get_training_batch():
    rand_idx = randint(0, num_samples - BATCH_SIZE - 1)
    batch_inputs = x_train[rand_idx: rand_idx + BATCH_SIZE]
    batch_labels = y_train[rand_idx: rand_idx + BATCH_SIZE]
    return batch_inputs, batch_labels[:, np.newaxis]

# Flag to determine training
train_word2vec = True

# Load saved data if available
if os.path.isfile('Word2VecXTrain.npy') and os.path.isfile('Word2VecYTrain.npy') and os.path.isfile('wordList.txt'):
    x_train = np.load('Word2VecXTrain.npy')
    y_train = np.load('Word2VecYTrain.npy')
    print("Loaded training matrices")

    with open("wordList.txt", "rb") as fp:
        vocabulary = pickle.load(fp)
    print("Loaded vocabulary list")

else:
    corpus_text, word_freq = load_dataset('conversationData.txt')
    print("Dataset loaded and processed")

    vocabulary = list(word_freq.keys())
    choice = input("Do you want to train your own Word2Vec embeddings? (y/n): ")

    if choice.lower() == 'y':
        x_train, y_train = build_training_data(word_freq, corpus_text)
        print("Training data created")
        np.save('Word2VecXTrain.npy', x_train)
        np.save('Word2VecYTrain.npy', y_train)
    else:
        train_word2vec = False

    with open("wordList.txt", "wb") as fp:
        pickle.dump(vocabulary, fp)

# Exit if not training Word2Vec
if not train_word2vec:
    sys.exit()

# Training setup
num_samples = len(x_train)
vocab_size = len(vocabulary)

session = tf.Session()

embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_DIM], -1.0, 1.0))
nce_weights = tf.Variable(tf.truncated_normal([vocab_size, EMBEDDING_DIM], stddev=1.0 / math.sqrt(EMBEDDING_DIM)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

input_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
label_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])

embed = tf.nn.embedding_lookup(embedding_matrix, input_placeholder)

loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=label_placeholder,
                   inputs=embed,
                   num_sampled=NEGATIVE_SAMPLES,
                   num_classes=vocab_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

session.run(tf.global_variables_initializer())

# Training loop
for step in range(NUM_TRAIN_STEPS):
    batch_inputs, batch_labels = get_training_batch()
    _, loss_val = session.run([optimizer, loss], feed_dict={input_placeholder: batch_inputs, label_placeholder: batch_labels})

    if step % 10000 == 0:
        print("Step:", step, "Loss:", loss_val)

# Save final embedding matrix
print("Saving trained embeddings...")
final_embeddings = embedding_matrix.eval(session=session)
np.save('embeddingMatrix.npy', final_embeddings)
