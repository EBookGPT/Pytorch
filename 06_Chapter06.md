# Chapter 6: Autograd in PyTorch

Welcome back, fellow PyTorch enthusiasts! In the last chapter, we learned about Variables and how they are the building blocks of PyTorch's computational graph. Variables enable us to perform gradient computations with ease. In this chapter, we are going to take things up a notch and dive deep into PyTorch's automatic differentiation engine - Autograd.

But before we do that, let me introduce you to our special guest for this chapter - Geoffrey Hinton. Geoffrey Hinton is a legendary computer scientist and a pioneer in deep learning. With numerous publications in machine learning and decades of experience, he is one of the most respected and influential figures in our industry. His contributions to the field, including the invention of backpropagation, have laid the foundation for much of what we do today in deep learning. We are honored to have him with us today.

With that being said, let's get started! In this chapter, we will examine the inner workings of Autograd and learn how it simplifies our lives by allowing us to compute gradients of complex functions. We will cover the following topics:

1. What is Autograd and how does it work?
2. How to create a dynamic computational graph?
3. How to compute gradients using Autograd?
4. Advanced example - building a neural network using Autograd.

So buckle up and let's learn some PyTorch!
# The Wizard of Autograd: A PyTorch Parable

Once upon a time, in the Land of Deep Learning, there lived a young aspiring data scientist named Dorothy. She dreamed of building the most powerful and accurate neural network, but she knew that the journey ahead would be full of challenges and obstacles.

One day, as she was wandering through the forest of algorithms, she stumbled upon a wise old man named Geoffrey. Geoffrey was known throughout the land as a great master of neural networks, and Dorothy was eager to learn from him.

Geoffrey took her under his wing and taught her the ways of PyTorch. He explained to her the importance of building a good computational graph and how it would be the foundation of all her deep learning models. Dorothy was fascinated by Geoffrey's knowledge and wanted to learn more.

Geoffrey then introduced her to Autograd, PyTorch's automatic differentiation engine. He told her that Autograd is what makes PyTorch so special because it eliminates the need to manually calculate gradients. With Autograd, Dorothy could focus solely on building her neural network and not worry about the complicated math behind it.

Dorothy was amazed by this and asked Geoffrey to show her how Autograd works. He conjured up a magical wand that allowed them to enter into the world of Autograd.

As they traversed through this world, Dorothy saw how Autograd builds and tracks the computational graph. The nodes of the graph represent the tensors and operations used in her neural network, while the edges represent the flow of data and gradients between them. Dorothy found this to be a brilliant solution to the problem of gradient computation and asked how to use it in her own models.

Geoffrey then taught Dorothy how to create a dynamic computational graph using Autograd. He explained that with a dynamic computational graph, she could build complex neural networks with variable input sizes and shapes. This would make her models more flexible and easier to use in real-world scenarios. Dorothy was ecstatic to learn this as she had always struggled when working with variable shapes in her models.

Finally, Geoffrey showed her how to use Autograd to compute gradients efficiently. Dorothy saw that with Autograd, all she had to do was perform forward propagation, and PyTorch would automatically calculate the gradients for her. She could then use these gradients to update the weights of her neural network during backpropagation.

As they returned to the forest of algorithms, Dorothy thanked Geoffrey for showing her the magic of Autograd. She now felt more confident in her ability to build powerful neural networks and knew that Autograd would be her secret weapon.

And so, Dorothy returned to her work, ready to face any challenge that lay ahead. With the knowledge of Autograd, she knew that she could build the most accurate and powerful neural network in all the land.
In this chapter, we learned about Autograd, PyTorch's automatic differentiation engine, and how it can be used to compute gradients of complex functions with ease. Here is a brief explanation of the code used to resolve our Wizard of Oz parable:

Firstly, the `requires_grad` attribute of a tensor must be set to `True`. This tells PyTorch to compute gradients with respect to that tensor during backpropagation.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

Next, we can perform forward propagation and compute some output.

```python
y = x.sum()
```

Then, we can call the `backward()` method on the output tensor to compute gradients.

```python
y.backward()
```

The gradients can be accessed through the `grad` attribute.

```python
print(x.grad)  # tensor([1., 1., 1.])
```

This will output the gradient of `y` with respect to `x`, which is simply a vector of ones.

We can also use Autograd to compute gradients of complex functions, such as building neural networks. Here's an example of how to build a simple neural network using Autograd:

```python
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 2)  # fully connected layer
        self.fc2 = nn.Linear(2, 1)  # fully connected layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create an instance of the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Forward propagation
inputs = torch.tensor([1.0, 2.0, 3.0])
output = net(inputs)

# Compute the loss
target = torch.tensor([0.5])  # a dummy target value
loss = criterion(output, target)

# Backward propagation
optimizer.zero_grad()
loss.backward()

# Update the weights
optimizer.step()
```

In this code, we define a simple neural network with two fully connected layers. We then instantiate the network, define the loss function and optimizer, and perform forward propagation on a dummy input tensor. We compute the loss and use Autograd to calculate the gradients with respect to the weights of the network. Finally, we use the optimizer to update the weights.

That's it for our brief explanation of the code used to resolve our Parable of the Wizard of Autograd. With the help of Autograd, you can focus on building and training powerful neural networks without worrying about the complicated math behind it. Happy coding!


[Next Chapter](07_Chapter07.md)