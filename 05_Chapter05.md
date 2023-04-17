# Chapter 5: Variables in PyTorch

Welcome to the next chapter on PyTorch! In the previous chapter, we learned about performing different operations on tensors. Now, in this chapter, we will delve into the world of variables in PyTorch. 

Variables are the objects responsible for building compute graphs in PyTorch. In simpler terms, variables keep track of the history of the events that they have been through—such as the operations performed on them—and produce incremental gradients for future operations. They are used to define and train neural networks within PyTorch, making it a favorite library of choice for deep learning practitioners. 

Conceptually, variables in PyTorch are very similar to Tensors. They hold data and support various operations that can be performed on them. However, there are fundamental differences in the way that variables are constructed and utilized. In particular, variables allow you to build a computational graph on the fly, which can help you calculate gradients in a flexible and efficient manner.

As we move forward in the chapter, we will cover the following topics:

1. Creating and initializing Variables 
2. Backpropagation with Variables
3. Computation Graphs and AutoGrad

So, are you ready to unravel the full potential of PyTorch? Let's dig into Variables!
# The Wizard of Oz Parable: Variables in PyTorch

Once upon a time, in the magical land of PyTorch, there was a young wizard named Dorothy. She was eager to learn about the secrets of neural networks and decided to embark on a journey to meet the great wizard of PyTorch, who lived in the Emerald City.

On the way to the Emerald City, Dorothy met a talking Tensor, who told her that to reach her destination, she needed to understand the power of Variables in PyTorch. The Tensor explained to her that Variables were objects that could help her perform different operations that would culminate in training her neural networks.

Dorothy was intrigued by this, and she asked for more information. The Tensor replied, "Unlike Tensors, which are static, Variables have the ability to store information about their history, such as the operations performed on them. This allows them to compute gradients quickly and efficiently, enabling the neural networks to learn and optimize their performance."

Dorothy realized that to become a true PyTorch wizard and build efficient neural networks, she needed to understand Variables. She thanked the Tensor for its advice and continued her journey towards the Emerald City.

Upon arriving at the Emerald City, Dorothy met the great wizard of PyTorch, who showed her how to create and initialize Variables. Under the wizard's guidance, Dorothy learned about backpropagation with Variables and how to use them to build computation graphs.

With the newfound knowledge of Variables, Dorothy was able to build better and more efficient neural networks. As she journeyed back home, she reflected on the lessons learned and how they could improve her future work in deep learning.

Just like Dorothy, mastering Variables in PyTorch can lead you to become a proficient wizard in deep learning. So, venture forth on this journey, and let the power of Variables guide your way towards efficient neural network development.
# Code for Variables in PyTorch

In the Wizard of Oz parable, we learned about the power of Variables in PyTorch. Now, let's take a closer look at the code used to implement and work with Variables in PyTorch.

## Creating and Initializing Variables

To create a Variable in PyTorch, we first need to import the necessary libraries and define our data. Once that's done, we can create a Variable by wrapping our tensor in the `torch.autograd.Variable()` function. Here's an example:

```
import torch

data = torch.Tensor([[1, 2], [3, 4]])
variable = torch.autograd.Variable(data, requires_grad=True)
```

In this example, we create a 2x2 tensor and wrap it in a Variable. We also set `requires_grad=True`, which tells PyTorch that we want to compute gradients with respect to this variable during backpropagation.

## Backpropagation with Variables

To perform backpropagation with Variables, we simply call the `backward()` function on our loss variable. PyTorch then computes gradients with respect to all the Variables in the computation graph. Here's an example:

```
import torch

# Define our model and loss
model = torch.nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()

# Prepare our data and Variables
data = torch.Tensor([[1, 2], [3, 4]])
inputs = torch.autograd.Variable(data, requires_grad=True)
labels = torch.autograd.Variable(torch.Tensor([[0], [1]]))

# Forward pass and loss calculation
outputs = model(inputs)
loss = loss_fn(outputs, labels)

# Backward pass and gradient calculation
loss.backward()
```

In this example, we define a simple linear model and mean squared error loss. We then wrap our data in Variables, perform the forward pass, and calculate the loss. Finally, we perform the backward pass by calling `loss.backward()`, which calculates the gradients with respect to all the Variables in the computation graph.

## Computation Graphs and AutoGrad

PyTorch uses an automatic differentiation package called AutoGrad to compute gradients. AutoGrad works by constructing a computation graph on-the-fly as you perform operations with Variables. This graph is then used to compute the gradients during backpropagation.

Here's an example of how to use AutoGrad to compute gradients:

```
import torch

# Define our Variables and operations
x = torch.autograd.Variable(torch.Tensor([2]), requires_grad=True)
y = x ** 2 + 3

# Perform backpropagation with AutoGrad
y.backward()

# Print the gradient
print(x.grad)
```

In this example, we create a variable `x` and perform an operation on it. We then call `backward()` on the resulting variable `y` and print the gradients with respect to `x`. PyTorch uses the computation graph generated by AutoGrad to efficiently calculate these gradients.

And that's a brief overview of the code used to implement and work with Variables in PyTorch. As you can see, Variables and AutoGrad provide a powerful and flexible framework for building and training neural networks.


[Next Chapter](06_Chapter06.md)