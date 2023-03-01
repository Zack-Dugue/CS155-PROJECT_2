# Import PyTorch Packages
import torch
from torch import nn
import torch.nn.functional as F
import csv
import pandas as pd
from surprise import prediction_algorithms as PA
from surprise import Dataset as DS
from surprise import Reader
import surprise.accuracy as ACC
import surprise as SP
from Visualize import visualize_factors
# Define the MF Model

def make_and_fit(train_data,test_data, K,reg, epochs,init_mean = 0,init_std_dev = 0.1):
    model = PA.SVD(n_factors=K, n_epochs=epochs, biased=True, init_mean=init_mean, init_std_dev=0.1, lr_all=0.005, reg_all=reg)
    model.fit(train_data)
    predictions = model.test(test_data, verbose=True)
    MSE = ACC.mse(predictions)
    print(f"MSE - {MSE}")
    return model

def load_data_for_surprise(path,m):
    data = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/data.csv")

    data = data.sample(frac=1).reset_index(drop=True)
    # split the data into training and test sets
    train = data[m:]
    test = data[:m]
    reader = Reader()
    train_set = DS.load_from_df(train,reader)
    test_set = DS.load_from_df(test,reader)
    return train_set, test_set

def param_search(train_data,test_data,n):

    param_dict = {"n_factors" : [10,20,30,40,50,75,100] , "n_epochs" : [65] ,
                  "lr_all" : [0.001 , .005, .01, 0.02], "reg_bu" : [0.005,0.01,0.02,0.05,.1,.2],"reg_bi" :[0.005,0.01,0.02,0.05,.1,.2],"reg_pu" :[0.005,0.01,0.02,0.05,.1,.2],"reg_qi" : [0.005,0.01,0.02,0.05,.1,.2]}
    model = SP.model_selection.search.RandomizedSearchCV(PA.SVD, param_dict, n_iter=n,  measures=['mse'], cv=None, refit=True, return_train_measures=True, n_jobs=-1,joblib_verbose=5)
    model.fit(train_data)
    print(f"Model Params = {model.best_params['mse']}")
    train_dataset = train_data.build_full_trainset().build_testset()
    print(f"E_IN:")
    predictions = model.test(train_dataset)
    MSE = ACC.mse(predictions)

    test_dataset = test_data.build_full_trainset().build_testset()
    print(f"E_OUT:")
    predictions = model.test(test_dataset)

    MSE = ACC.mse(predictions)

    return model

def experiment(m,n):
    train_data, test_data = load_data_for_surprise("CS155-PROJECT_2/data/data.csv",m)
    model = param_search(train_data,test_data,n)
    model = model.best_estimator['mse']
    return model


if __name__ == "__main__":
    model = experiment(250,20)
    U = model.pu
    V = model.qi
    data = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/data.csv")
    # visualize_factors(data,U,V)
