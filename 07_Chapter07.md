# Chapter 7: Neural Networks using PyTorch 

Welcome back! In the previous chapter, we talked about how autograd works in PyTorch, allowing us to easily compute gradients and perform backward propagation. 

Now, it's time to dive into the wonderful world of neural networks using PyTorch. And we have a special guest with us - none other than the "Godfather of deep learning" himself, Geoffrey Hinton. 

Geoffrey Hinton is a world-renowned computer scientist and cognitive psychologist, known for his pioneering work in deep learning and neural networks. He has won numerous awards and accolades for his contributions, including the Turing Award, considered the "Nobel Prize of computing".

In this chapter, we will explore how PyTorch makes it easy to build and train neural networks. We'll start by introducing the basics of neural networks, their architecture and the math behind them. Then, we'll dive into PyTorch's neural network package - torch.nn - and see how we can use it to define and train complex neural networks.

Hinton, in his recent paper [1], has proposed a radical new approach to training neural networks called "Transformers". This approach aims to replace the traditional recurrent neural networks with attention models. It has been demonstrated that "Transformers" achieve state-of-the-art results on several natural language processing tasks, including machine translation and language modeling. 

So let's buckle up and get ready to explore the exciting and ever-evolving world of neural networks using PyTorch, with our special guest, Geoffrey Hinton!

References: 
1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.A.N., Gomez, A.N., Kaiser, L.U., & Polosukhin, I. (2017). Attention Is All You Need. arXiv preprint arXiv:1706.03762.
# The Wizard of Oz Parable: Neural Networks using PyTorch

Once upon a time, in the magical land of PyTorch, there lived a young girl named Dorothy who wanted to build the best neural network and win the hearts of everyone in her village. She had heard of Geoffrey Hinton's remarkable work in deep learning and wanted to learn from the master himself.

So, she set off on a journey along with her companions - a brave lion who feared nothing, a scarecrow with no brains, and a tin man who had no heart. They ventured through mountains and forests until they reached the magnificent castle, where Geoffrey Hinton resided.

Upon meeting Hinton, Dorothy expressed her desire to build a powerful neural network, and the wise wizard agreed to teach her the ways of PyTorch.

Hinton began by explaining the concept of neural networks—their architecture, the math behind them, and the latest advancements in this field. He showed her how the intricate web of layers and nodes could be optimized to perform complex tasks, such as image classification, speech recognition, and language modeling.

With PyTorch, Hinton said, building neural networks was easier than ever. Dorothy could use PyTorch's neural network package, torch.nn, to define and train complex models with just a few lines of code. The package offered a wide range of layers, including convolutional, recurrent, and fully connected layers, as well as activation functions like ReLU, Sigmoid, and Tanh, which could be used to build different types of neural networks, including CNNs, RNNs, and Transformers.

Hinton also showed her how to use PyTorch's built-in optimizers, such as SGD, Adam, and RMSprop, to train her models and adjust their weights and biases, gradually improving their performance over time.

Dorothy listened to Hinton intently and was amazed by the power of PyTorch. She and her companions began building their own neural networks using PyTorch's tools and techniques, and soon they were able to recognize images, process speech, and generate meaningful text with their models.

But their journey wasn't over yet. They continued to learn from Hinton, exploring new concepts, such as regularization, dropout, and batch normalization, which helped them refine their models even further.

Finally, after months of hard work, they presented their models to Hinton, who was impressed by their creations. He said that their neural networks were some of the best he had seen in recent times.

Dorothy and her companions returned to their village, proud of what they had accomplished. They showed their models to the people and won the hearts of everyone in the village. They continued to experiment with new ideas and techniques, always keeping in mind the lessons they had learned from Geoffrey Hinton and the magical land of PyTorch.

And thus, they lived happily ever after, building remarkable neural networks and making a name for themselves in the world of deep learning.

References: 
1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.A.N., Gomez, A.N., Kaiser, L.U., & Polosukhin, I. (2017). Attention Is All You Need. arXiv preprint arXiv:1706.03762.
# The Code: Building Neural Networks using PyTorch

Now that we've heard the Wizard of Oz parable about building neural networks with PyTorch, let's take a look at some code examples that demonstrate how we can translate the lessons learned into practice.

We'll start with a basic example of how to define and train a simple neural network using PyTorch. 


```
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=64, shuffle=True)

# Set up the model
net = Net()

# Set up the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
```

This code demonstrates how to define a basic neural network, in this case for an image classification task using the familiar MNIST dataset. We define a two-layer fully connected neural network, with a ReLU activation applied to the hidden layer and a softmax activation applied to the output layer. We then use the SGD optimizer with a momentum of 0.5 and a learning rate of 0.01, and define the cross-entropy loss function.

Next, we load the data and set up the model, optimizer, and loss function. Then we train the model for 10 epochs, using mini-batches of size 64, and gradually reduce the loss.

This is just a basic example of how to use PyTorch to build neural networks. There are many more techniques and optimizations that we can use to improve the performance of our networks, including regularization, dropout, and batch normalization. But by starting with these simple steps, we can take the first step to building more powerful and sophisticated neural networks with PyTorch.

Happy learning!


[Next Chapter](08_Chapter08.md)