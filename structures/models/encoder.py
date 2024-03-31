import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

  def __init__(self,layers=1,
               number_heads = 1,
               input_size=10,
               hidden_size=5,
               time_steps=10,
               dropout=0.2,
               device='cpu'):
    super().__init__()
    head_size = hidden_size // number_heads
    self.time_steps = time_steps
    self.device = device
    self.dim = input_size
    self.embedding_table = nn.Embedding(input_size, hidden_size, device=device)
    self.pos_embedding_table = nn.Embedding(time_steps, hidden_size, device=device)
    encoder_layer = nn.TransformerEncoderLayer(hidden_size,
                                                number_heads,
                                                dim_feedforward=4*hidden_size,
                                                activation='relu',
                                                norm_first=True,
                                                bias=True,
                                                batch_first=True,
                                                device=device )
    self.transformerEncoder = nn.TransformerEncoder(encoder_layer,layers, enable_nested_tensor=False)
    # self.blocks = nn.Sequential(*[Block(hidden_size, head_size, dropout, number_heads) for i in range(layers)])
    self.final_layer_norm = nn.LayerNorm(hidden_size, device=device)
    self.linear_output = nn.Linear(hidden_size, input_size, device=device)

  def forward(self, x_input, y_targets):

    time_array = torch.arange(self.time_steps).to(self.device)
    pos = self.pos_embedding_table(time_array)
    x = x_input + pos
    x = self.transformerEncoder(x)
    x = self.final_layer_norm(x)
    x = self.linear_output(x)
    logits = x[:,-1,:].view(x.shape[0],1, self.dim)
    B,T,C = logits.shape
    logits = logits.view(B*T, C)
    y_targets = y_targets.view(B*T).long()

    loss = F.cross_entropy(logits, y_targets)
    return logits, loss

  def info(self,layers=True):
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    if layers:
      print("Layers:")
      for name, param in self.named_parameters():
          print(f"Layer: {name}, Size: {param.size()}, Parameters: {param.numel()}")


"""Old implmentation from attention is all you need"""
# class Head(nn.Module):

#   def __init__(self, input_size, head_size, dropout, linearBias=False):
#     super().__init__()
#     self.head_size = head_size
#     self.keys = nn.Linear(input_size, head_size, bias=linearBias)
#     self.queries = nn.Linear(input_size, head_size, bias=linearBias)
#     self.values = nn.Linear(input_size, head_size, bias=linearBias)
#     self.dropout = nn.Dropout(dropout)

#   def forward(self, x):
#     k = self.keys(x)
#     q = self.queries(x)
#     v = self.values(x)

#     weights = (q @ torch.transpose(k, 1,2))*(self.head_size)**-0.5
#     weights = F.softmax(weights, dim=1)
#     weights = self.dropout(weights)
#     logits = weights @ v
#     # print("logits shape", logits.shape)
#     return logits


# class MultiHeadAttention(nn.Module):

#   def __init__(self,input_size, head_size, dropout, number_heads):
#     super().__init__()
#     self.heads = nn.ModuleList([Head(input_size, head_size, dropout) for i in range(number_heads)])
#     self.layer = nn.Sequential(
#         nn.Linear(input_size, input_size),
#         nn.Dropout(dropout)
#     )

#   def forward(self, x):
#     multi_head_attention = [head(x) for head in self.heads]
#     # print("multi_head_attention shape  ", multi_head_attention)
#     logits = torch.cat(multi_head_attention, -1)
#     logits = self.layer(logits)
#     return logits

# class FeedForwardNetwork(nn.Module):

#   def __init__(self, input_size, dropout):
#     super().__init__()
#     self.feed_forward = nn.Sequential(
#         nn.Linear(input_size, input_size*4),
#         nn.ReLU(),
#         nn.Linear(input_size*4, input_size),
#         nn.Dropout(dropout)
#     )

#   def forward(self, x):
#     logits = self.feed_forward(x)
#     return logits

# class Block(nn.Module):

#   def __init__(self, input_size, head_size, dropout, number_heads):
#     super().__init__()
#     self.heads = MultiHeadAttention(input_size,head_size,dropout, number_heads)
#     self.feed_forward = FeedForwardNetwork(input_size, dropout)
#     self.layer1_norm = nn.LayerNorm(input_size)
#     self.layer2_norm = nn.LayerNorm(input_size)

#   def forward(self, x):
#     x = x + self.heads(self.layer1_norm(x))
#     logits = x + self.feed_forward(self.layer2_norm(x))

#     return logits
