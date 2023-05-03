import torch
import torch.optim as optim

# Define some example parameters and gradients
params = [torch.randn(2, 2, requires_grad=True), torch.randn(2, 2, requires_grad=True)]
gradients = [torch.randn(2, 2), torch.randn(2, 2)]

# Define an optimizer and compute some updates
optimizer = optim.SGD(params, lr=0.1)
for i in range(5):
    optimizer.zero_grad()
    loss = sum((params[j] * gradients[j]).sum() for j in range(len(params)))
    loss.backward()
    optimizer.step()

# Print the number of updates made by the optimizer
print(optimizer.step)
