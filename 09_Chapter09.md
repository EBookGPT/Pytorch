# Chapter 9: Recurrent Neural Networks in PyTorch

Greetings, journeyers in the land of PyTorch! In the previous chapter, we explored the ins and outs of Convolutional Neural Networks (CNNs). We learned how these powerful networks can be used to analyze visual data and extract meaningful features. In this chapter, we are going to dive into a new type of neural network: Recurrent Neural Networks (RNNs).

While CNNs are excellent at recognizing patterns in static images, RNNs are designed to work with sequences of data. They can be used to model sequences of almost any kind - from human speech to stock prices to the trajectory of a particle in physics. With the ability to analyze sequences in this way, the potential applications of RNNs are vast and exciting.

In this chapter, we will explore the architecture and inner workings of RNNs. We will walk through the process of building an RNN in PyTorch, and learn how to train our network to recognize patterns in sequences of data. By the end of this chapter, you will have a strong understanding of the theory behind RNNs and the practical skills to implement them in your own work.

So let's grab our wizard hats and dive in!
# The Wizard of Oz and the Magical Sequences

Once upon a time in the land of PyTorch, there was a wise wizard named Oz. Oz had spent many years studying the magic of neural networks, and had become an expert in the art of pattern recognition.

One day, as he strolled through the forest, he came across a group of travelers who were lost and in need of his help. The travelers were trying to analyze a sequence of data - the electrical activity in a person's brain - but they were having trouble finding patterns in the data.

"Please, Oz," they begged. "Can you help us discover the meaning behind these sequences?"

Oz could see the struggle on their faces, and he knew exactly what to do. He pulled out his wand and began to weave a spell. With a wave of his wand and some magic words muttered under his breath, Oz conjured up a Recurrent Neural Network (RNN).

"You see," explained Oz, "an RNN is the perfect tool for analyzing sequences of data. It has the ability to remember past inputs and use that information to predict future outputs."

The travelers watched in awe as Oz worked his magic. He taught them how to build an RNN in PyTorch, and how to use it to recognize patterns in the brain data. With each passing moment, the RNN became more accurate, and soon the travelers were able to make groundbreaking discoveries about the brain's activity.

As they said their goodbyes, the travelers couldn't help but feel grateful to Oz for his help. "How can we ever repay you?" they asked.

Oz simply smiled and said, "Spread the knowledge! Teach others about the magic of PyTorch and Recurrent Neural Networks. Share the power of pattern recognition with the world."

And so the travelers went on their way, armed with the knowledge and skills they had learned from Oz. They knew that with their newfound abilities, they could go out into the world and analyze all kinds of sequences - from heartbeats to musical notes - and find the meaning behind the patterns.

And who knows? Maybe one day they too would become wizards like Oz, passing on their knowledge and helping others change the world with the magic of data analysis.
# Resolving the Wizard of Oz Parable with PyTorch

In the story of the Wizard of Oz and the Magical Sequences, we saw how Oz used the power of PyTorch and Recurrent Neural Networks (RNNs) to help a group of travelers analyze a sequence of brain data. Now, let's take a closer look at the code that Oz used to build and train the RNN.

### Building the RNN

First, Oz imported the necessary libraries:

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

Then, he defined the RNN class:

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
```

The RNN class takes three parameters: `input_size`, `hidden_size`, and `output_size`. It initializes two linear layers (`i2h` and `i2o`) and a LogSoftmax function, and defines the `forward` function for processing input and generating output. The `initHidden` function initializes the hidden state with a tensor of zeroes.

### Training the RNN

Next, Oz defined the training function:

``` python
def train_rnn(rnn, criterion, optimizer, input_tensor, target_tensor):
    hidden = rnn.initHidden()

    optimizer.zero_grad()

    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])

    loss.backward()

    optimizer.step()

    return output, loss.item()
```

The `train_rnn` function takes the RNN, criterion, optimizer, input sequence, and target sequence as input. It initializes the hidden state, sets the optimizer gradients to zero, and loops through each element of the input sequence. On each loop, it generates an output using the RNN, calculates the loss using the criterion, and updates the hidden state. Finally, it backpropagates the loss and updates the optimizer.

### Putting it all together

To train the RNN, Oz used a loop that called the `train_rnn` function for each epoch:

``` python
rnn = RNN(n_letters, 128, n_categories)
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)

n_epochs = 1000
print_every = 50
plot_every = 100

current_loss = 0
all_losses = []

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train_rnn(rnn, criterion, optimizer, line_tensor, category_tensor)
    current_loss += loss

    # Print epoch number, loss, and example
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
```

In this loop, Oz defined the RNN, criterion, and optimizer, as well as the number of epochs and how often to print and plot the loss. He then looped through each epoch, generating a random training example, calling the `train_rnn` function, and printing out the loss and accuracy at regular intervals.

And that, my friends, is how Oz used the power of PyTorch and RNNs to solve the problem of the Magical Sequences. With this knowledge, you too can wield the power of sequence analysis and change the world!


[Next Chapter](10_Chapter10.md)