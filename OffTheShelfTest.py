import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
import pandas as pd
from OffTheShelf import MF

def load_data(m):
    movies = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/movies.csv")
    data = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/data.csv")
    # randomly shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    # split the data into training and test sets
    train = data[:m]
    test = data[m:]
    train = th.tensor(train)
    test = th.tensor(test)

    return train, test

def train(train,test,model,epochs = 30, batch_size = 32,lr = .001):
    train_dataset = data_utils.TensorDataset(train.float(), train.float())
    data = data_utils.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
    optimizer = th.optim.SGD(model.parameters(),lr)
    for epoch in range(epochs):
        avg_loss = 0
        counter = 0
        for (x,_) in data:
            optimizer.zero_grad()
            loss = model.loss(x)
            avg_loss += loss
            counter += 1
            loss.backward()
            optimizer.step()
        print(f"epoch : {epoch} - avg_loss: {avg_loss/counter}")

    return model

def experiment(m):
    load_data(m)
    MF(992 - m, 1500 - m , )