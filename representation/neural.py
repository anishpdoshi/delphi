import os
import torch
import numpy as np
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pickle

import pandas as pd
from base_learner import BaseLearner
from nn_to_smt import layers_to_smt

def FeedForwardNN(quantized=False):
    layers = []
    if quantized:
        layers.append(quantization.QuantStub())
    layers = [
        nn.Linear(7, 50),
        nn.ReLU(),
        nn.Linear(50, 1),
        nn.ReLU(),
    ]
    if quantized:
        layers.append(quantization.DeQuantStub())
    
    return nn.Sequential(*layers)

def train_model(model, loader, num_epochs=10, lr=0.001):
    epochs = num_epochs
    criterion = nn.MSELoss()
    # create your optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for i in range(epochs):
        l = 0
        for input, target in loader:
            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update
            l += loss.item()

        print(f'Epoch {i} loss: {l/50}')
        scheduler.step(l)
    
    return model

class NeuralLearner(BaseLearner):
    def __init__(self, interface):
        super().__init__(interface)
        # if interface[1] == 'BV'
        self.nn = FeedForwardNN(quantized=False)
        self.standard_scaler = None

    def get_paths(self, location):
        nn_path = location + '_nn.pt'
        scaler_path = location + '_scaler.pkl'
        return nn_path, scaler_path

    def load_from(self, location):
        # Load the NN and StandardScaler
        nn_path, scaler_path = get_paths(location)
        self.nn.load_state_dict(torch.load(nn_path))
        self.standard_scaler = pickle.load(open(scaler_path,'rb'))

    def train(self, examples, update_pretrained=False, train_args={}):
        X, y = list(zip(examples))
        X, y = np.array(X), np.array(y)
        
        lr = train_args.get('lr', 0.1)

        # Refit a standard scaler if not pretrained
        if not self.standard_scaler:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(X)

        X_std = self.standard_scaler.transform(X)

        batch_size = min(10, len(examples))
        dataset = TensorDataset(torch.Tensor(X_std), torch.Tensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        train_model(self.nn, loader, lr)
        
        if update_pretrained:
            nn_path, scaler_path = self.get_paths(update_pretrained)
            torch.save(self.nn.state_dict(), nn_path)
            pickle.dump(self.standard_scaler, open(scaler_path, 'wb'))
    
    def run(self, inputs):
        X = np.array(inputs)
        if self.standard_scaler:
            X = self.standard_scaler.transform(X)
        else:
            print('WARN no standard scaler initialized')
        
        batch_size = min(10, len(examples))
        dataset = TensorDataset(torch.Tensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        preds = []
        for inp in dataloader:
            preds.extend(model(inp).flatten().tolist())
        
        return preds

    def to_smt2(self):
        return layers_to_smt(self.nn)