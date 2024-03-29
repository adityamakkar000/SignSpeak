import torch
import torch.nn as nn
import pandas
import torch.nn.functional as F

# hyper parameters

vocab_size = 10
n_emb = 5
n_layers = 1
time_steps = 10

#encoder layer
data = pandas.read_csv('./Data.csv')
#prepare classes
y = data['word']
stoi  = {s:i for i,s in enumerate(sorted(set(y)))}
encode = lambda s: stoi[s]
y = torch.tensor([encode(s) for i,s in enumerate(y)])
# y = F.one_hot((y.view(y.shape[0], 1)),10) # don't use one hot with cross entropy loss

#prepare data
x = torch.tensor(data.iloc[:,2:time_steps*n_emb + 2].to_numpy())
x = torch.nan_to_num(x).float()
x = x.view(x.shape[0], -1, n_emb)
print(x.dtype, " ", y.dtype)

print(x.shape, " ", y.shape)

class Head(nn.Module):

  def __init__(self):
    super().__init__()
    self.head_size = n_emb // n_layers
    self.keys = nn.Linear(n_emb, self.head_size, bias=False)
    self.queries = nn.Linear(n_emb, self.head_size, bias=False)
    self.values = nn.Linear(n_emb, self.head_size, bias=False)

  def forward(self, x):
    k = self.keys(x)
    q = self.queries(x)
    v = self.values(x)

    weights = (q @ torch.transpose(k, 1,2))*(self.head_size)**-0.5
    weights = F.softmax(weights, dim=1)
    logits = weights @ v
    return logits

class Encoder(nn.Module):

  def __init__(self):
    super().__init__()
    # self.embedding_table = nn.Embedding(vocab_size, n_emb)
    self.pos_embedding_table = nn.Embedding(time_steps, n_emb)
    self.self_attention = Head()
    self.final_layer_norm = nn.LayerNorm(n_emb)
    self.linear_output = nn.Linear(n_emb, vocab_size)

  def forward(self, x_input, y_targets):
    pos = self.pos_embedding_table(torch.arange(time_steps))
    x = x_input + pos
    x = self.self_attention(x)
    x = self.final_layer_norm(x)
    x = self.linear_output(x)
    logits = x[:,-1,:].view(x.shape[0],1,vocab_size)
    B,T,C = logits.shape
    logits = logits.view(B*T, C)
    y_targets = y_targets.view(B*T).long()

    loss = F.cross_entropy(logits, y_targets)
    return logits, loss

test_x = x[0].view(1,10,5)
test_y = y[0].view(1,1).int()

model = Encoder()
parameters = model.parameters()
s = sum([p.nelement() for p in parameters])
print(s)

logits,loss = model(test_x, test_y)
print(logits, " ", loss)
