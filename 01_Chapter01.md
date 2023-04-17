# Introduction to PyTorch

If you are reading this chapter, chances are you are already familiar with the buzzwords in the field of machine learning and deep learning. You must have heard about PyTorch - the powerful and versatile framework for building and training deep neural networks. PyTorch is one of the most widely used deep learning frameworks today due to its simplicity, flexibility, and ease of use. 

PyTorch was developed by Facebook AI Research (FAIR) and is an open-source project. The framework combines the power of two of the most important areas in computer science today, Python programming and deep learning. PyTorch has continued to grow in popularity because it provides excellent support for debugging neural networks and offers an easy-to-use API that supports powerful GPU acceleration.

In this chapter, we will explore some of the fundamental concepts of PyTorch, including tensors, automatic differentiation, and neural network modules. We will also learn how to utilize PyTorch to build our first deep neural network. So let's buckle up and embark on our journey of learning PyTorch!
# The Wizard of PyTorch

Once upon a time, in the magical land of Machine Learning, there lived a young data scientist named Dorothy. Dorothy was as curious as she was ambitious and always wanted to learn more about the mysterious ways of artificial intelligence. One day, she heard of a great and powerful wizard who was said to know everything about the art of deep learning. They called him...the Wizard of PyTorch!

Dorothy set out on a journey to find the Wizard and learn from him. She traveled through dense jungles of pre-processing and preprocessing, over the steep hills of gradient descent, and across vast oceans of transfer learning. Finally, after a long and tiring journey, she arrived at the Wizard's castle.

The Wizard welcomed her with open arms and showed her the wonders of PyTorch. He talked about tensors, which were the basic building blocks of PyTorch, and showed her how to perform operations using them. He introduced her to the concept of automatic differentiation, which, according to him, was one of the most powerful tools for optimizing neural networks. Dorothy was amazed and couldn't wait to learn more.

As the days went by, the Wizard taught her how to create and train neural networks using PyTorch. Dorothy learned how to use PyTorch to build simple neural networks with fully connected layers and train them on classification tasks. The Wizard even showed her how to use Convolutional Neural Networks for image classification and Recurrent Neural Networks for time-series analysis.

Dorothy was thrilled with all she had learned from the Wizard of PyTorch. She felt confident and empowered to take on even more challenging problems in the world of machine learning. She thanked the Wizard for his invaluable teachings and set out on her journey back home.

As she reached the end of her journey, Dorothy realized that PyTorch was not just a tool for building powerful neural networks, but also a community of like-minded individuals eager to explore and push the boundaries of deep learning. And so, armed with her newfound knowledge and the magic of PyTorch, Dorothy set out to change the world...one tensor at a time.
## Explanation of PyTorch Code Used

In the Wizard of Oz parable, the protagonist Dorothy uses PyTorch to build and train different neural networks. Let's take a closer look at some of the PyTorch code she used in her journey.

### Tensors

The Wizard introduced Dorothy to Tensors, which are the basic building blocks of PyTorch. In PyTorch, a tensor is a generalization of a matrix to arbitrary dimensions, and is often used to represent numerical data. 

In PyTorch, creating a tensor is as simple as running the following code:

```
import torch

# Create a tensor of zeros with shape (3, 5)
x = torch.zeros(3, 5)
```

This creates a tensor `x` with dimensions 3 x 5, where all the elements are zero.

### Automatic Differentiation

To optimize neural networks, the Wizard taught Dorothy about the powerful technique of automatic differentiation. In PyTorch, automatic differentiation is achieved through the `autograd` package, which tracks the operations performed on tensors and allows for easy computation of gradients.

Here's a simple example to demonstrate how to create a tensor with the `requires_grad` attribute and compute gradients:

```
import torch

# Create a tensor with requires_grad set to True
x = torch.tensor([2., 3.], requires_grad=True)

# Perform some operations with x
y = x * 3
z = y.mean()

# Compute gradients with respect to x
z.backward()

# The gradients are now stored in x.grad
print(x.grad)
```

### Neural Network Modules

Finally, the Wizard showed Dorothy how to use PyTorch to build and train neural networks. In PyTorch, this is achieved through the use of `nn.Module`, a high-level API that makes constructing and training neural networks straightforward. 

Here's a simple example of how to define and train a simple feedforward neural network with PyTorch:

```
import torch
import torch.nn as nn

# Define a simple feedforward neural network
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU(x)
        x = self.fc2(x)
        return x

# Instantiate the network and define the loss function and optimizer
model = MyNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Train the network
for epoch in range(num_epochs):
    for input, target in dataset:
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

In this example, we defined a simple neural network with two fully connected layers and trained it on a dataset using the Cross-Entropy loss function and stochastic gradient descent optimizer.

With these powerful tools at her disposal, Dorothy was able to become a master of PyTorch and accomplish her goals in the magical world of machine learning!


[Next Chapter](02_Chapter02.md)