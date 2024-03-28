from torcheval.metrics.functional import multiclass_f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
import numpy as np


# upload data
def data_upload(model, collection):
  cms = np.array(model.confusion_matrixs).tolist()
  for i in model.final_results:
    loss, accuracy, f1score, avg_f1score, cateogrial_accuracy = i
    f1score = f1score.tolist()
    data  = {'model': model.type, 'loss': loss, 'accuracy': accuracy,
             'avg_f1score': avg_f1score,'cateogrical_f1score': f1score,
             'confusion_matrix': cms}
    collection.insert_one(data)



class getModel():

  def __init__(self,type=False,stacks=1,dense=True):
    self.model =

class model(nn.Module):

  def __init__(self, nin, nout):
    super.__init__()
      per.__init__()

  def forward(self, x):


data = pandas.read_csv('./data/Data.csv')

#prepare classes
y = data['word']
stoi  = {s:i for i,s in enumerate(sorted(set(y)))}
encode = lambda s: stoi[s]
y = torch.tensor([encode(s) for i,s in enumerate(y)])
# y = F.one_hot((y.view(y.shape[0], 1)),10) # don't use one hot with cross entropy loss

#prepare data
x = torch.tensor(data.iloc[:,2:].to_numpy())
x = torch.nan_to_num(x)


# prepare classes
data = pandas.read_csv('./Data.csv')


y = data['word']
stoi  = {s:i for i,s in enumerate(sorted(set(y)))}
encode = lambda s: stoi[s]
y = torch.tensor([encode(s) for i,s in enumerate(y)])
print(y)

# y = F.one_hot((y.view(y.shape[0], 1)),10)

x = data.iloc[:,2:52].to_numpy()
print(data.iloc[:,2:])
x = torch.tensor(x, dtype = torch.float32)
x = torch.nan_to_num(x)
x = x.view(x.shape[0], -1, 5)
print(x[0,:5])

torch.manual_seed(1)
class LSTMStacked(nn.Module):

  def __init__(self, batch_first=True, layers=1, dense_layer=False):
    super().__init__()
    self.layer1 = nn.LSTM(5, 64, num_layers = 2, batch_first = batch_first)
    self.layer2 = nn.Linear(64,64)
    self.layer3 = nn.Tanh()
    self.layer4 = nn.Linear(64,10)

  def forward(self, x):
    self.inter_output, self.inter_h = self.layer1(x)
    self.layer2out = self.layer2(self.inter_h[-1])
    self.layer3out = self.layer2out.tanh()
    self.out = self.layer4(self.layer3out)

    return self.out

model1 = LSTMStacked(True,1)
model2 = LSTMStacked()

output1 = model1(x)

print(output1.shape)

epochs = 10000
optimizer = torch.optim.AdamW(model1.parameters(),lr=1e-3)

for i in range(epochs):

  x_output = model1(x)
  loss = F.cross_entropy(x_output,y)

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  f1 = multiclass_f1_score(x_output, y, num_classes=10, average='macro')

  if(i % 100 == 0):
    print(i, " ",loss.item(), " ", f1)

  optimizer.step()


