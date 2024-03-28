import torch
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

