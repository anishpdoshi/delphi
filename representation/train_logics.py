import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import pandas as pd

hidden_size = 10

class Dataset(torch.utils.data.Dataset):

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.Tensor(X)
      self.y = torch.Tensor(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

def create_nn(hidden_size):
    return nn.Sequential(
        nn.Linear(7, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
        nn.ReLU(),
    )

def preds_from(model, dataloader):
    preds = []
    for inp, targ in dataloader:
        preds.extend(model(inp).flatten().tolist())
    return np.array(preds)

def train_model(model, train_dataloader, lr=0.001):

    epochs = 100
    criterion = nn.MSELoss()
    # create your optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    for i in range(epochs):
        l = 0
        for input, target in train_loader:
            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update
            l += loss.item()
            # print(loss.item())
        print(f'Epoch {i} loss: {l/50}')
        scheduler.step(l)
    
    return model

# learn_representation
# neural.py
# basething.py
# symbolic_regression.py


# need to be able to run this from command line
# frontend when passed --neural
#     goes to neural runner
#     create net (with or w/o quantization)?
#     if --pretrained is passed in
#         load net from file
#     given example, 
#         learn from that example
#         print(nn_to_smt(example))

# when passed --symb or --dt
#     either 
#         take in csv path, and run adhoc pretraining then
#     take in --logics flag, and load 



if __name__ == '__main__':
    
    
    df = pd.read_csv('logics_samples.csv')
    X = df.iloc[:,1:8].values
    y = df['dragForce'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f'Rescaled with mean {scaler.mean_} and var {scaler.var_}')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

    model = create_nn(hidden_size)
    model_filename = f'logics_nn_{hidden_size}.pt'
    if os.path.exists(model_filename):
        print(f'{model_filename} exists, loading from there and not training')
        model.load_state_dict(torch.load(model_filename))
    else:
        epochs = 100
        criterion = nn.MSELoss()
        # create your optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        for i in range(epochs):
            l = 0
            for input, target in train_loader:
                # in your training loop:
                optimizer.zero_grad()   # zero the gradient buffers
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()    # Does the update
                l += loss.item()
                # print(loss.item())
            print(f'Epoch {i} loss: {l/50}')
            scheduler.step(l)
        
        torch.save(model.state_dict(), model_filename)
    
    model.eval()
    
    with torch.no_grad():
        y_train_preds = preds_from(model, train_loader)
        y_test_preds = preds_from(model, test_loader)
    
    print(f'Train mean squared error: {mean_squared_error(y_train_preds, y_train)}')
    print(f'Pred mean squared error: {mean_squared_error(y_test_preds, y_test)}')

    print(f'Train mean absolute error: {mean_absolute_error(y_train_preds, y_train)}')
    print(f'Pred mean absolute error: {mean_absolute_error(y_test_preds, y_test)}')

    print(f'Train mape error: {mean_absolute_percentage_error(y_train_preds, y_train)}')
    print(f'Pred mape error: {mean_absolute_percentage_error(y_test_preds, y_test)}')

    print(f'Train max error: {max_error(y_train_preds, y_train)}')
    print(f'Pred max error: {max_error(y_test_preds, y_test)}')
