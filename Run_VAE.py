#!/usr/bin/env python
# Run_roboVAE.py
# By Shawn Beaulieu
# July 21st, 2017

import ast
import math
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from VariationalBayes import VAE
from tensorflow.examples.tutorials.mnist import input_data

def generate_labels(index, length, population):                        
    label = [0]*length                                                  
    label[index] = 1                       
    labels = [label]*population                                        
    return(labels) 

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return(a[p], b[p])

def generate_batch(X, y, batch_size):
    num_instances = X.shape[0]
    randSample = random.sample(range(num_instances), batch_size)
    x_batch = X[randSample, :]
    y_batch = y[randSample]
    return(x_batch, y_batch)

def preserve(data, filename):

    df_data = pd.DataFrame(data)

    try:
        # Read old file to remove duplicates
        old_file = pd.read_csv(filename, sep=",", header=None)
        # DF is 1x784, where final column = fitness
        os.system("rm {0}".format(filename))
        old_file = old_file.append(df_data)
        old_file.to_csv(filename, sep=",", header=None, index=None)
    except:
        # If no such file exists (new experiment) create it:
        df_data.to_csv(filename, sep=",", header=None, index=None)


def Recursive_Pass(data, labels, model, passes):
    # Add timestamp for saving images
    idx = np.random.choice(range(data.shape[0]))
    copy = data[idx]
    label = labels[idx]
    plt.imshow(copy.reshape(28,28), cmap='Greys')
    plt.show()
    for p in range(passes):
        copy = model.Reconstruct(copy.reshape(1,784), label.reshape(1,10))
        plt.imshow(copy.reshape(28,28), cmap='Greys')
        plt.show()

def Random_Sample(model, sample_size, conv, timestamp):
   
    if conv == False:
        label_dict = {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        }

        filename = "Opt_ACCORDION_GAUSSD_100E_2000B_at_{0}.csv".format(timestamp)

        for s in range(sample_size):
            label = [0]*10
            idx = np.random.choice(range(10))
            label[idx] = 1
            label = np.array(label).reshape(1,10)
            sample = model.Generate(pop_size=1, labels=label)
            df = pd.DataFrame(sample)

            try:
                df_samples = df_samples.append(df) 
            except:
                df_samples = df             

            label_output = "Perceived Label: {0}".format(label_dict[idx])

            print(label_output)
            with open("Labels_{0}".format(filename), "a+") as labelfile:
                labelfile.write(label_output)
                labelfile.write("\n")

            plt.imshow(sample.reshape(28,28), cmap='Greys')
            plt.show()
    
    else:

        for s in range(sample_size):
            sample = model.Generate(pop_size=1, labels=0)
            plt.imshow(sample.reshape(28,28), cmap='Greys')
            plt.show()

    df_samples.to_csv("Samples_{0}".format(filename), header=None, index=None)

def Reconstruct(model, data, labels, sample_size):
    for s in range(sample_size):
        recon = model.Reconstruct(data[s].reshape(1,-1), labels[s].reshape(1,-1))
        plt.imshow(data[s].reshape(28,28), cmap='Greys')
        plt.show()
        plt.imshow(recon.reshape(28,28), cmap='Greys')
        plt.show()

def Train(data, labels, blueprint, convolutions, hyperparameters,
          training_epochs, display_step, filename, timestamp, new_graph=True):

    model = VAE(blueprint, hyperparameters, convolutions, meta_graph=None, new_graph=True)
    # Initialize cost:
    cost = {'t-1': 0.0, 't': 0.0}
    for epoch in range(training_epochs):
        cost['t-1'] = cost['t']
        cost['t'] = 0.0
        num_batches = int(data.shape[0]/hyperparameters['batch_size'])
        #if epoch % 5 == 0:
        #    hyperparameters['dropout_rate'] -= hyperparameters['dropout_rate']/5
        for iteration in range(num_batches):
            X_batch, y_batch = generate_batch(data, labels, hyperparameters['batch_size'])
            new_cost = model.Fit(X_batch, hyperparameters['dropout_rate'], y_batch)/hyperparameters['batch_size']
            cost['t'] += new_cost
            # Archive cost:
            #with open("Cost_Opt_ACCORDION_GAUSSD_100E_2000B_at_{0}.csv".format(timestamp), "a+") as cost_file:
            #    cost_file.write(str(new_cost))
            #    cost_file.write(str("\n"))
        
        cost['t'] /= num_batches
        change = cost['t'] - cost['t-1']
        if epoch % display_step == 0:
            print("Epoch {0}: Cost = {1}, Change = {2}".format(epoch, cost['t'], change))

    # SAVE GRAPH
    if new_graph:
        model.Save(filename)

    return(model)

if __name__ == '__main__':
    
    # Data is 28x28 greyscal images
    fashion_mnist = data = input_data.read_data_sets('data/fashion', one_hot=True)
    X_train, y_train = unison_shuffle(fashion_mnist.train.images, fashion_mnist.train.labels)
    X_test, y_test = unison_shuffle(fashion_mnist.test.images, fashion_mnist.test.labels)
    
    blueprint = [1000, 1500, 750, 375, 4]
    convolutions = [(4, 4, 1, 8), (4, 4, 8, 16)]
    convolutions = 0

    HYPERPARAMETERS = {

        "batch_size": 4000,
        "regularizer": 1E-6,
        "learning_rate": 3E-4,
        "dropout": True,
        "dropout_rate": 0.50,
        "num_classes": 10
    }

    training_epochs = 10
    timestamp = datetime.now().strftime(r"%y%m%d_%H%M")
    model = Train(X_train, y_train, blueprint, convolutions, HYPERPARAMETERS,training_epochs=training_epochs, \
                            display_step=1, filename="Fashion_MNIST_VAE", timestamp=timestamp, new_graph=True)

    if sys.argv[1] == "sample":
        Random_Sample(model, sample_size=25, conv=False, timestamp=timestamp)
    elif sys.argv[1] == "recursion":
        Recursive_Pass(X_test, y_test, model, 50)
    elif sys.argv[1] == "recon":
        Reconstruct(model, X_test, y_test, 10)
