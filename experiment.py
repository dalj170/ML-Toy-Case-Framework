import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import SimpleDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from datagenerator import Generator, Dynamics
from models import *

# default parameters can be modified pretty easily, no real basis for a 1e-4 learning rate or any of the other settings.
# changing batch sizes or n_data may cause some issues though, as the data is split 80%/20% train/test, and both of
# those resulting numbers must be divisible by the batch_size_train and batch_size_val
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size_train', type=int, default=100)
parser.add_argument('--batch_size_val', type=int, default=100)
parser.add_argument('--t_span', type=float, default=None)
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--n_data', type=int, default=500)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# generating data using functions from datagenerator.py
f = Dynamics
data = Generator(f, args.dt, args.n_data, args.t_span)

# works for an n x 2 matrix but would not for a larger dimensional one
x = data[:, 0]
y = data[:, 1]


# making the dataset and randomly splitting for test/validation
dataset = SimpleDataset(x, y)
train_data, val_data = random_split(dataset, [int(len(x)*.80), int(len(y)*.20)])

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size_train, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size_val, shuffle=True)

lr = args.lr
n_epochs = args.n_epochs

# define model and optimizer. Adam seems to perform better than SGD
model = EncoderDecoder(args.batch_size_train).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# define loss and training functions (functions in models.py)
loss_fn = loss_calc2
train_step = make_train_step(model, loss_fn, optimizer)

losses = []
val_losses = []

for epoch in range(n_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # the unsqueezing is done to make the x and y (which is x(t+1)) vectors from vectors that only have one
        # dimension into two dimensional matrices with one dimension being 1. Also done to get orientation correct for
        # matrix multiplication.
        x_batch = x_batch.unsqueeze(1)
        y_batch = y_batch.unsqueeze(1)
        loss = train_step(x_batch, y_batch, args.dt)
        losses.append(loss)

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            x_val = x_val.unsqueeze(1)
            y_val = y_val.to(device)
            y_val = y_val.unsqueeze(1)

            model.eval()
            yhat = model(x_val, args.dt)

            val_loss = loss_fn(x_val, y_val, args.dt, yhat, model)
            val_losses.append(val_loss.item())

    if epoch % 25 == 0:
        print("Epoch: {:04d} | train loss: {:0.6f} | val loss: {:0.6f}".format(epoch, losses[len(losses) - 1],
                                                                               val_losses[len(val_losses) - 1]))

# 0 -> 2 -> 4 in terms of layers. 0.weight, 0.bias etc
print(model.ev.float())

# TODO: Status update:
"""
I think I've generalized it better, but now it's harder to tell. 
"""
