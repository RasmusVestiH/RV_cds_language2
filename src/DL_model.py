# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report


# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# matplotlib
import matplotlib.pyplot as plt

def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
def main():
    filepath = '../data/Game_of_Thrones_Script.csv'
    dataframe = pd.read_csv(filepath)

    # get the values in each cell; returns a list
    sentences = dataframe['Sentence'].values
    season = dataframe['Season'].values

    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                        season, 
                                                        test_size=0.25, 
                                                        random_state=42)

    vectorizer = CountVectorizer()

    # First we do it for our training data...
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then we do it for our test data
    X_test_feats = vectorizer.transform(X_test)

    # We must convert the season to numbers as labels for the words embedding
    labels = pd.factorize(season)[0]

    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]

    # initialize tokenizer
    tokenizer = Tokenizer(num_words=5000)
    # fit to training data
    tokenizer.fit_on_texts(X_train)

    # tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    # inspect
    print(X_train[2])
    print(X_train_toks[2])

    # max length for a doc
    maxlen = 100

    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                                padding='post', # sequences can be padded "pre" or "post"
                                maxlen=maxlen)
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                               padding='post', 
                               maxlen=maxlen)

    # define embedding size we want to work with
    embedding_dim = 50

    # initialize Sequential model
    model = Sequential()
    # add Embedding layer
    model.add(Embedding(input_dim=vocab_size,     # vocab size from Tokenizer()
                        output_dim=embedding_dim, # user defined embedding size
                        input_length=maxlen))     # maxlen of padded docs
    # add Flatten layer
    model.add(Flatten())
    # Add Dense layer; 10 neurons; ReLU activation
    model.add(Dense(10, 
                    activation='relu'))
    # Add prediction node; 1 node with sigmoid; approximates Logistic Regression
    model.add(Dense(1, 
                    activation='sigmoid'))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # print summary
    model.summary()

    history = model.fit(X_train_pad, y_train,
                        epochs=20,
                        verbose=False,
                        validation_data=(X_test_pad, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    predictions = model.predict(X_test_pad, batch_size = 10)
    print(classification_report(y_test, predictions.argmax(axis=1)))

if __name__ == "__main__":
    main()