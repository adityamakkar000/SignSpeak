import torch
import torch.nn as nn

# Define the TDNN model
class TDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Define layers
        self.tdnn_layer = nn.Linear(input_dim, 60)
        self.hidden_layer1 = nn.Linear(60, 60)
        self.hidden_layer2 = nn.Linear(60, 70)
        self.output_layer = nn.Linear(70, output_dim)
        
    def forward(self, x):
        # Apply TDNN layer
        x = torch.relu(self.tdnn_layer(x))
        
        # Apply first hidden layer
        x = torch.relu(self.hidden_layer1(x))
        
        # Apply second hidden layer
        x = torch.relu(self.hidden_layer2(x))
        
        # Apply output layer
        x = self.output_layer(x)
        return x

# Example usage
input_dim = 60  # Input feature dimension
output_dim = 3  # Output dimension

# Create an instance of the TDNN model
model = TDNN(input_dim, output_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Sample input data (adjust as needed)
input_data = torch.randn(10, input_dim)  # 10 samples, each with 60 features

# Forward pass
output = model(input_data)

# Compute the loss
target = torch.randn(10, output_dim)  # Example target values (adjust as needed)
loss = criterion(output, target)

epochs = 5

for i in range(epochs):
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

# Make predictions (for new input data)
new_input_data = torch.randn(5, input_dim)  # 5 new samples
predictions = model(new_input_data)

print(predictions)