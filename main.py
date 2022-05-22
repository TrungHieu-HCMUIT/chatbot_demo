import pickle
import json
from tempfile import TemporaryFile
from urllib import response
import nltk
import numpy
import tflearn
import tensorflow
import random
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops

print('\n=====DEMO======\n')
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    sentences_by_word_x = []
    sentences_by_word_y = []

    # Data pre-processing
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            sentences_by_word_x.append(wrds)
            sentences_by_word_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # Stemming: The process of changing words into root words
    words = sorted(list(set(words))) # Remove duplicate words

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for i, sentence in enumerate(sentences_by_word_x):
        bag = []

        stemmed_sentence = [stemmer.stem(s) for s in sentence]

        for w in words:
            if w in stemmed_sentence:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(sentences_by_word_y[i])] = 1
        
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8) # 1st hidden layer with 8 neurons
net = tflearn.fully_connected(net, 8) # 2nd hidden layer with 8 neurons
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if (se == w):
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print('Start talking with the chatbot (type quit to stop)!')
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        result_index = numpy.argmax(results)
        tag = labels[result_index]

        if results[result_index] > 0.7:
            for intent in data["intents"]:
                if intent["tag"] == tag:
                    responses = intent["responses"]
            print(random.choice(responses))
        else:
            print("Sorry, I didn't get that. Try again")

chat()
