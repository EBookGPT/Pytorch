# Chapter 4: Operations on Tensors

Welcome back to our PyTorch book! In the previous chapter, we learned about the basics of Tensors in PyTorch. Now it's time to explore operations on Tensors.

Operations on Tensors are very useful in many machine learning applications, such as image processing, natural language processing, and even music creation. In this chapter, we will dive deeper into the world of Tensors and explore some of the operations that PyTorch provides. 

To make things extra special, we have invited a special guest, Jeremy Howard, to help us with this chapter! Jeremy is a highly respected deep learning researcher who has co-founded fast.ai and previously worked at Kaggle and Google. His contributions to the field have been recognized by Forbes’ "30 under 30" in technology and Wired UK's "25 people shaping the future."

Jeremy will guide us through some of the more advanced techniques and applications of Tensor operations, so get ready for some exciting stuff! 

Let's get started! 

## Common Operations on Tensors

PyTorch provides a wide range of operations for manipulating Tensors. Here are some of the most common:

### Arithmetic Operations

PyTorch Tensors support all the standard arithmetic operators, such as addition, subtraction, multiplication, and division. Let's see how these operations are applied on our Tensors.

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Addition
print("Addition: ", tensor1 + tensor2)

# Substraction
print("Substraction: ", tensor1 - tensor2)

# Multiplication
print("Multiplication: ", tensor1 * tensor2)

# Division
print("Division: ", tensor1 / tensor2)
```

### Matrix Multiplication

One of the most important operations on Tensors is matrix multiplication. PyTorch provides the `torch.mm()` function to perform matrix multiplication. 

```python
# Matrix multiplication
print("Matrix multiplication: ", torch.mm(tensor1, tensor2))
```

### Reshaping Tensors

Sometimes data needs to be reshaped so that it can be properly processed. For example, we may need to convert a 2D matrix into a 1D vector, or vice versa. And PyTorch provides a handy function for this: `torch.view()`

```python
# Reshaping matrix
tensor3 = tensor1.view(1, 4)
print("After reshaping matrix: ", tensor3)

# Converting matrix into vector
tensor4 = tensor3.view(4)
print("After converting matrix into vector: ", tensor4)
```

## Conclusion

We've only scratched the surface of the rich set of operations that can be performed on PyTorch Tensors. With Jeremy's help, we've seen how arithmetic operations, matrix multiplication, and reshaping can be performed on Tensors. 

In the next chapter, we'll continue our exploration of PyTorch by learning about neural networks, something that Jeremy is a true expert in!
# Chapter 4: Operations on Tensors - The Wizard of Oz Parable

Once upon a time, there was a young data scientist named Dorothy who dreamed of creating state-of-the-art machine learning models. She had heard of a powerful tool called PyTorch that could help her accomplish this goal. But she didn't know where to start.

One day, as she was struggling with her latest project, she heard a knock at the door. She opened it to find a man with a friendly smile, who introduced himself as Jeremy Howard, a well-known deep learning researcher.

"Hello, Dorothy! I heard you've been wanting to learn more about Tensor operations in PyTorch. I'm here to guide you through your journey!" exclaimed Jeremy.

Dorothy was thrilled and invited him inside. Jeremy sat down with her and began explaining the basics of Tensor operations.

"There are many operations that PyTorch provides for manipulating Tensors, such as arithmetic operations, matrix multiplication, and reshaping. These operations can help you process data and create powerful models," explained Jeremy.

Dorothy was eager to learn more and asked Jeremy to show her some examples. 

"Of course, let's start with arithmetic operations!" said Jeremy. He took out a pen and paper and drew two matrices, one with the values [[1, 2], [3, 4]] and another with values [[5, 6], [7, 8]].

"Add these two matrices together and you get the matrix," said Jeremy, scribbling the answer. "Similarly, you can perform subtraction, multiplication, and division operations on Tensors."

Dorothy was amazed at how simple the arithmetic operations were in PyTorch. She was eager to try them out for herself.

"Wait, it gets even better," said Jeremy. "Matrix multiplication is one of the most important operations on Tensors. PyTorch provides efficient functions to perform matrix multiplication, such as torch.mm()."

He demonstrated how to perform matrix multiplication on the matrices. Dorothy was fascinated by how quick and easy it was.

"And finally, Tensors can be reshaped, too," Jeremy concluded. "Sometimes data may need to be transformed so that it can be properly processed. PyTorch provides the torch.view() function that allows you to reshape Tensors in many ways."

Jeremy demonstrated how to reshape a matrix and convert it to a 1D vector.

Dorothy was grateful for Jeremy's help, and she was now confident that she could apply these operations to her own projects.

"Thank you so much, Jeremy! You are like the Wizard of Oz, showing me the way to using PyTorch efficiently," she said gratefully.

Jeremy smiled, happy to be of service. "Remember, the real magic is in the learning and creation of something valuable. Keep on learning and creating, Dorothy!"

And with that, Dorothy continued on her journey to becoming a skilled deep learning practitioner, with Jeremy Howard by her side.
In this chapter, Dorothy gets help from Jeremy Howard to learn about Tensor operations in PyTorch. They explore some of the most common operations like arithmetic operations, matrix multiplication, and tensor reshaping.

To demonstrate these concepts, some code samples are shown. 

First, arithmetic operations are shown on two tensors named `tensor1` and `tensor2`:

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Addition
print("Addition: ", tensor1 + tensor2)

# Substraction
print("Substraction: ", tensor1 - tensor2)

# Multiplication
print("Multiplication: ", tensor1 * tensor2)

# Division
print("Division: ", tensor1 / tensor2)
```

These operations are applied element-wise to the corresponding values in the two tensors.

Then, matrix multiplication between two tensors named `tensor1` and `tensor2` is demonstrated:

```python
# Matrix multiplication
print("Matrix multiplication: ", torch.mm(tensor1, tensor2))
```

PyTorch provides built-in functions that can perform matrix multiplication efficiently, such as torch.mm().

Next, tensor reshaping is demonstrated:

```python
# Reshaping matrix
tensor3 = tensor1.view(1, 4)
print("After reshaping matrix: ", tensor3)

# Converting matrix into vector
tensor4 = tensor3.view(4)
print("After converting matrix into vector: ", tensor4)
```

This reshapes a tensor from one shape to another specified shape, given in the argument to the `view()` function.

Through these code examples, Dorothy learned how to perform Tensor operations in PyTorch with ease, thanks to Jeremy's help!


[Next Chapter](05_Chapter05.md)