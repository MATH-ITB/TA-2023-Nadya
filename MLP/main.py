#!/usr/bin/env python3

import mlp
import pandas as pd

if __name__ == "__main__":
    df_train = pd.read_csv('MLP\mnist_data\mnist_train.csv').to_numpy()
    df_test= pd.read_csv('MLP\mnist_data\mnist_test.csv').to_numpy()
    
    x_train, y_train = df_train[:,1:],df_train[:,0]
    x_test, y_test = df_test[:,1:],df_test[:,0]

    # normalize data
    x_train = x_train/x_train.max()
    x_test = x_test/x_test.max()

    model = mlp.MLP(x_train,y_train,x_test,y_test,L=1,N_l=128)
    model.train(batch_size=8,epochs=25,lr=1.0)

