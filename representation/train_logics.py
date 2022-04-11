import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

import pandas as pd

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

def net():
    return nn.Sequential(
        nn.Linear(7, 50),
        nn.ReLU(),
        nn.Linear(50, 1),
        nn.ReLU(),
    )

def createDataset(csv_file):
    df = pd.read_csv(csv_file)
    X = []
    y = []
    for i, row in df.iterrows():
        row = list(row[1:])
        X.append(row[:7])
        y.append([row[7]])
    return Dataset(X, y)

if __name__ == '__main__':
    logics_net = net()
    if os.path.exists('logics_nn.pt'):
        logics_net.load_state_dict(torch.load('logics_nn.pt'))
    else:
        dataset = createDataset('logics_samples.csv')
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

        epochs = 50
        criterion = nn.MSELoss()
        # create your optimizer
        optimizer = optim.Adam(logics_net.parameters(), lr=0.001)
        for i in range(epochs):
            l = 0
            for input, target in trainloader:

                # in your training loop:
                optimizer.zero_grad()   # zero the gradient buffers
                output = logics_net(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()    # Does the update

                l += loss.item()
                # print(loss.item())
            print(f'Epoch {i} loss: {l/50}')
        
        torch.save(logics_net.state_dict(), 'logics_nn.pt')