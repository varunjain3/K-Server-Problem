import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import RNN_Model
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split




class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, values):
        'Initialization'
        self.values = torch.tensor(values)
        self.labels = torch.tensor(values)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.values[index]
        y = self.labels[index]

        return X, y


class Train_Memory_Model(pl.LightningModule):
    def __init__(self, max_pages=100, embedding_size=128, hidden_dim=128, n_layers=1, p=0.4):
        super().__init__()

        intermediate_layer = int(max_pages//2)

        # Memory Model
        self.max_pages = max_pages
        self.memory_model = RNN_Model(
            max_pages, embedding_size, hidden_dim, n_layers, p)
        # Freeze the weights of the embedding layer
        self.memory_model.input_embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(hidden_dim, intermediate_layer)
        self.bn = nn.BatchNorm1d(intermediate_layer)
        self.fc2 = nn.Linear(intermediate_layer, max_pages)

    def forward(self, x):
        batch_size = x.size()[0]
        x.unsqueeze_(-1)
        hidden_dim = self.memory_model.hidden_dim
        out_score, hidden = self.memory_model(x)

        out_space = self.fc1(hidden.view(batch_size, hidden_dim))
        out_space = self.bn(out_space)
        out_space = self.fc2(out_space)
        out_score = F.log_softmax(out_space, dim=-1)
        return out_score, hidden

    def training_step(self, batch):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x_hat, hidden = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch,batch_idx):
        # validation_step defines the validation loop. It is independent of forward
        x, y = batch
        x_hat, hidden = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=0.1)
        return optimizer


'''
# we want to freeze the fc2 layer this time: only train fc1 and fc3
net.fc2.weight.requires_grad = False
net.fc2.bias.requires_grad = False

# passing only those parameters that explicitly requires grad
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

# then do the normal execution of loss calculation and backward propagation

#  unfreezing the fc2 layer for extra tuning if needed
net.fc2.weight.requires_grad = True
net.fc2.bias.requires_grad = True

# add the unfrozen fc2 weight to the current optimizer
optimizer.add_param_group({'params': net.fc2.parameters()})
'''

if __name__ == '__main__':
    # pl set seed
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_PAGES = 10000

    temp = list(range(MAX_PAGES))

    dataset = Dataset(temp)
    train, val = random_split(dataset, [int(MAX_PAGES*0.9), int(MAX_PAGES*0.1)])

    batch_size = 512
    shuffle = True
    pin_memory = True
    num_workers = 4
    train = DataLoader(train, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, persistent_workers=True)
    val = DataLoader(val, batch_size=batch_size, shuffle=False,num_workers=num_workers, persistent_workers=True)


    autoencoder = Train_Memory_Model(max_pages=MAX_PAGES)
    trainer = pl.Trainer(accelerator="gpu",devices=2, max_epochs=100,accumulate_grad_batches={5: 3, 10: 20},log_every_n_steps=5)
    trainer.fit(autoencoder, train,val)
