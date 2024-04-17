import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import lightning as L
import pandas
from sklearn.model_selection import StratifiedKFold


class getDataset(Dataset):
  """ pytorch Dataset object generation

  Turn a tensor into a dataset
  e.g Take in ASL measuremnts tensors and tensor labels and turn it into an iterable
  dataset to be used by Pytorch DataLoader

  """
  def __init__(self, x, y):
    """Assign tensors to self states"""

    self.x = x
    self.y = y

  def __len__(self):
    """Length is the number of examples in a dataset (0-indexed demension = a single gesture)"""

    return self.x.shape[0]

  def __getitem__(self, idx):
    """get item and it's label with given index idx"""

    sample = {
      "measurement": self.x[idx],
      "label": self.y[idx]
    }

    return sample


class ASLDataModule(L.LightningDataModule, Dataset):
  """Lightning Module for use of trainer"""

  def __init__(self,
               kfold,
               splits,
               seed,
               time_steps,
               n_emb,
               batch_size,
               shuffle=True,
               dir='./data/test2.csv',
               ):
    super().__init__()

    self.data_dir = dir

    self.seed = seed
    self.splits = splits
    self.kfold = kfold
    self.time_steps = time_steps
    self.n_emb = n_emb

    # intialize generator every time class is intialized to have same indexes for train split
    self.generator = torch.Generator(device='cpu').manual_seed(self.seed)

    # params to be used for train DataLoader
    self.params = {
      'batch_size': batch_size,
      'shuffle': shuffle,
      'generator': self.generator
    }

  def prepare_data(self):
    """prepare data as tensors"""

    data = pandas.read_csv(self.data_dir, dtype={"word": "string"})
    y = data['word'] # extract classes
    number = [str(i) for i in range(20,30)]
    indexes = [i for i,s in enumerate(y) if s not in number] # remove numbers
    y = [s for i,s in enumerate(y) if s not in number]
    stoi  = {s:i for i,s in enumerate(sorted(set(y)))} # assign indexes to each possible class (a - z, 1-10)
    encode = lambda s: stoi[s] # inline function to covert character to class number
    y = torch.tensor([encode(s) for i,s in enumerate(y)]) # encode letters into words

    """
    x is a 3-dim tensor
    dim-0 gives the different measurments (keep all for all samples)
    dim-1 takes values from index-2 onward (0:2 is the _id value + word labels)
    dim-1 extends self.time_steps * self.n_emb as for each time step there is self.n_emb measumrents (5 for each finger)
    exporting to csv compresses 3D data into 2D hence, self.time_steps * self.n_emb
    """
    x = data.iloc[:,2:self.time_steps*self.n_emb + 2].to_numpy()

    x = [torch.tensor(i)[~torch.isnan(torch.tensor(i))] for i in x] # remove nan values
    x = pad_sequence([i for i in x], batch_first=True, padding_value=0) # pad values to same sequence length
    x = x.view(x.shape[0], self.time_steps, self.n_emb).float() # seperate into B x time_steps x n_emb
    x = x[indexes] # remove numbers from labels

    print(x.shape, "   ", y.shape)


    self.length = x.shape[0]

    """create kfold split and assign the train and valdatasets to the current fold"""
    kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.seed)

    for i, (train, test) in enumerate(kfold.split(x.cpu(),y.cpu())):
      if i == self.kfold:
        self.train_dataset = getDataset(x[train], y[train])
        self.val_dataset = getDataset(x[test], y[test])
        break

  def train_dataloader(self):
    """called when trainer.fit() is used"""

    return DataLoader(self.train_dataset, **self.params)

  def val_dataloader(self):
    """called when trainer.val() is used in training cycle"""
    return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))
