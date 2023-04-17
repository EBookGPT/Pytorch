# Chapter 3: Tensors in PyTorch

Welcome back to our PyTorch journey! In the previous chapter, we learned how to install and set up PyTorch. Now, it's time to dive deeper into the foundational element of PyTorch: **Tensors**.

Tensors are multi-dimensional arrays that are the building blocks of PyTorch. They are similar to NumPy arrays but are designed to be computed on with GPUs, making them ideal for deep learning algorithms. Tensors can be one-dimensional like a vector, two-dimensional like a matrix, or have higher dimensions.

In this chapter, we’ll take a closer look at tensors and explore how they are used in PyTorch. We’ll cover the following topics:

- Creating tensors
- Indexing and slicing tensors
- Basic tensor operations
- Broadcasting in PyTorch tensors

We’ll also discuss the benefits of using tensors and their impact on deep learning research. By the end of this chapter, you’ll have a solid understanding of tensors and be ready to explore their use in more advanced PyTorch concepts.

Before we dive in, it's important to note that familiarity with NumPy arrays is a prerequisite for understanding tensors. If you’re not familiar with NumPy, we recommend brushing up on your NumPy knowledge before diving into this chapter.

Get ready to level up your PyTorch skills with tensors!
# The Parable of the PyTorch Tensors

Once upon a time, in a far-off land, lived a young wizard named Oz. Oz had always dreamed of creating powerful spells with a magical force, and his dream came true when he discovered a mystical artifact: the PyTorch Tensor.

The PyTorch Tensor was a powerful tool that allowed Oz to manipulate data with ease. It was a multi-dimensional array that could store and process large amounts of data at lightning speeds. The Tensor fascinated Oz, and he set out to learn all about its powers.

As Oz delved deeper into the world of Tensors, he discovered that they had many dimensions, much like the different magic spells he had learned. He could create Tensors that were one-dimensional or had many dimensions, depending on what he needed.

"Just like magic spells, Tensors can be customized to suit any task." Oz thought to himself.

He learned how to create Tensors with ease using the PyTorch library. Just like using a magic wand, he could create a Tensor with a single command. Here is an example of creating a Tensor in PyTorch:

```
import torch

# creating a 1-dimensional Tensor with a list
my_tensor = torch.tensor([1, 2, 3])

# creating a 2-dimensional Tensor with a nested list
my_tensor_2d = torch.tensor([[1, 2], [3, 4], [5, 6]])
```

But creating a Tensor was just the beginning. Oz soon discovered that by slicing and indexing Tensors, he could extract specific values or sections of the Tensor that he needed. With this power, he could perform complex calculations on large datasets with ease.

As Oz became more experienced with Tensors, he learned about broadcasting. Just like casting a spell that affects an entire group of people, broadcasting allowed him to apply a single value or operation to a section, or the entire Tensor, without having to write out each calculation.

In the end, Oz was able to accomplish incredible feats with PyTorch Tensors. His journey taught him the value of Tensors in deep learning research, and how they could be used to create powerful magic.

Just like Oz, the power of Tensors is available to anyone who is willing to learn. By unleashing the power of PyTorch, anyone can manipulate large amounts of data and create complex models that can change the world.
Throughout the parable, we used code snippets to illustrate various concepts related to PyTorch Tensors. Here’s an explanation of each of them:

Creating a PyTorch Tensor:
```python
import torch

# creating a 1-dimensional Tensor with a list
my_tensor = torch.tensor([1, 2, 3])

# creating a 2-dimensional Tensor with a nested list
my_tensor_2d = torch.tensor([[1, 2], [3, 4], [5, 6]])
```
In this code, we are importing PyTorch and creating two Tensors, a 1D Tensor with three elements and a 2D Tensor with three rows and two columns. We use the `torch.tensor()` function to create the Tensors from a list or a nested list.

Indexing and slicing a PyTorch Tensor:
```python
# creating a 2-dimensional Tensor with a nested list
my_tensor_2d = torch.tensor([[1, 2], [3, 4], [5, 6]])

# indexing and slicing tensors
print(my_tensor_2d[0])       # get the first row
print(my_tensor_2d[0][1])    # get the second element in the first row
print(my_tensor_2d[:, 1])    # get the second column
```
In this code, we are demonstrating how to index and slice Tensors with the example of the 2D Tensor `my_tensor_2d`. We can use indexing to extract specific rows and columns, and slicing to extract sections of the Tensor. In the above example, the first print statement retrieves the first row of the Tensor, the second print statement retrieves the second element in the first row, and the third print statement retrieves the entire second column.

Broadcasting a PyTorch Tensor:
```python
# creating two Tensors
my_tensor = torch.tensor([1, 2, 3])
scalar = 2

# broadcasting the scalar value to the Tensor
print(my_tensor * scalar)    # multiply each element by 2
```
In this code, we are creating a Tensor `my_tensor` with three elements and a scalar `scalar` equal to 2. We then multiply the Tensor by the scalar using the `*` operator. Broadcasting allows us to apply a scalar value to the entire Tensor without having to write out each calculation.

With these code examples, you now have a better understanding of how PyTorch Tensors work and how they can be created, indexed, sliced and broadcast. These concepts form the building blocks for more advanced PyTorch concepts such as neural networks and deep learning.


[Next Chapter](04_Chapter04.md)