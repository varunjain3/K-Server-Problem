from matplotlib.pyplot import axis
from torch import nn
import torch
import torch.nn.functional as F

# make sure to use the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class InputEmbedding(nn.Module):

#     def __init__(self, max_pages=100,embedding_space = 128, ):
#         super().__init__()
#         self.max_pages = max_pages
#         self.input_embedding = nn.Embedding(max_pages, embedding_space) # Embedding layer
#         self.embedding_dropout = nn.Dropout(p)
#         # self.batch_norm = nn.BatchNorm1d(embedding_space,axis=1)
#         self.layer_norm = nn.LayerNorm(embedding_space)


#     def forward(self, x):
#         x = self.input_embedding(x)
#         x = self.embedding_dropout(x)
#         x = self.layer_norm(x)
#         return x


class RNN_Model(nn.Module):
    def __init__(self, max_pages=100, embedding_size=128, hidden_dim=512, n_layers=1, p=0.4):
        super().__init__()

        

        # Layers for Embedding
        self.max_pages = max_pages
        self.input_embedding = nn.Embedding(max_pages, embedding_size) # Embedding layer
        self.embedding_dropout = nn.Dropout(p)
        self.layer_norm = nn.LayerNorm(embedding_size)

        # Layers for RNN
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(embedding_size, hidden_dim, n_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_dim, max_pages)

    def forward(self, x):
        batch_size,n = x.size()
        hidden_dim = self.hidden_dim
        
        # Passing through embedding layer
        x = self.input_embedding(x)
        x = self.embedding_dropout(x)
        x = self.layer_norm(x)

        # Passing through RNN
        out,hidden = self.rnn(x)

        return out, hidden
    

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden




if __name__ == '__main__':

    # lstm = nn.LSTM(3, 3,)  # Input dim is 3, output dim is 3
    # inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # # initialize the hidden state.
    # hidden = (torch.randn(1, 1, 3),
    #         torch.randn(1, 1, 3))
    # for i in inputs:
    #     # Step through the sequence one element at a time.
    #     # after each step, hidden contains the hidden state.
    #     out, hidden = lstm(i.view(1, 1, -1), hidden)

    batch_size = 16
    n = 10
    x = torch.tensor(torch.randint(0,100,(batch_size,n,)),device=device)
    # model = InputEmbedding().to(device)
    # x = model(x)

    model1 = RNN_Model().to(device)
    x = model1(x)

    # print(model(x).shape)