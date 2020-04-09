import torch
import torch.nn as nn
# this wont be generalizable code, it will be limited to 1 ev


class EncoderDecoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        # parameter for eigenvalue
        self.ev = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

        # I'm assuming the encoder should be able to learn the entire representation of the eigenfunction, including
        # the constant C and the eigenvalue. The eigenvalue defined separately is used to step phi(x(t)) forward to
        # phi(x(t+1))
        # no specific reason for the dimensions of these linear parts

        self.encoder = nn.Sequential(nn.Linear(1, 10),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(10, 10),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(10, 1))

        self.decoder = nn.Sequential(nn.Linear(1, 10),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(10, 10),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(10, 1))

    def forward(self, x, dt):
        # returns forward step, i.e. xt+1
        # attempting to do this as generally as possible, hoping the network learns something resembling the form of the
        # needed transformation to phi, Ce^lambda*f(x)
        phi = self.encoder(x)
        g = phi * torch.exp(-dt * torch.exp(self.ev))
        y = self.decoder(g)

        return y
        # return torch.exp(-(dt * torch.tensor(1.1).cuda()**self.ev)) * self.w * self.m * x
        # access weights for loss with model.enc1.weight.float()
        # access ev and x with model.state_dict()['ev']


# function to perform a training step


def make_train_step(model, loss_fn, optimizer):
    # make_train_step is (i think) just a constructor that lets train_step be called more easily
    # steps through training loop
    def train_step(x, y, dt):
        model.train()

        # yhat = model(x, dt)
        yhat = model(x, dt)

        loss = loss_fn(x, y, dt, yhat, model)

        # torch.autograd.backward() computes backward pass of entire graph
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        return loss.item()

    return train_step


# loss functions


def loss_calc(x_train, y_train, dt, y_hat, ef, ev):
    """ unused """
    return ((y_train - y_hat) ** 2).mean() \
           + ((ef * y_train - torch.exp(-dt * torch.exp(ev)) * ef * x_train) ** 2).mean()


def loss_calc2(x_t, y_t, dt, y_hat, model):
    """ loss function used to optimize model. Three main terms, the first checks the prediction accuracy, the second
    checks the reconstruction accuracy of the encoder and decoder (i.e. x = phi^-1(phi(x))), and the last makes sure the
    measure space representation of x(t+1) is the same as the measure space representation of x(t) advanced one time
    step.
    This is a little gross since I had to be creative in order to get the weights out of the model. Two helper functions
    are defined below, where the learned weights can be applied to any input.
    """
    return ((y_t - y_hat) ** 2).mean() + ((x_t - phi_inv(model, phi(model, x_t))) ** 2).mean() \
        + ((phi(model, y_t) - phi(model, x_t) * torch.exp(-dt * torch.exp(model.state_dict()['ev']))) ** 2).mean()

    # Three loss functions
    # first is for prediction, x_t+1 - phi^-1(K*phi(x))
    # second is for autoencoder, tests \phi^-1(\phi(x)) = x
    # third is for linear dynamics? \phi(x_t+1) = K*phi(x)


def phi(model, x):
    """ applies the weights learned in the model to x for loss purposes"""
    x = torch.mm(x, torch.transpose(model.encoder[0].state_dict()['weight'], 0, 1)) + model.encoder[0].state_dict()[
        'bias']
    x = torch.relu(x)
    x = torch.mm(x, torch.transpose(model.encoder[2].state_dict()['weight'], 0, 1)) + model.encoder[2].state_dict()[
        'bias']
    x = torch.relu(x)
    x = torch.mm(x, torch.transpose(model.encoder[4].state_dict()['weight'], 0, 1)) + model.encoder[4].state_dict()[
        'bias']
    return x


def phi_inv(model, x):
    """ converts the measure space representation of x back into normal representation"""
    x = torch.mm(x, torch.transpose(model.decoder[0].state_dict()['weight'], 0, 1)) + model.decoder[0].state_dict()[
        'bias']
    x = torch.relu(x)
    x = torch.mm(x, torch.transpose(model.decoder[2].state_dict()['weight'], 0, 1)) + model.decoder[2].state_dict()[
        'bias']
    x = torch.relu(x)
    x = torch.mm(x, torch.transpose(model.decoder[4].state_dict()['weight'], 0, 1)) + model.decoder[4].state_dict()[
        'bias']
    return x
