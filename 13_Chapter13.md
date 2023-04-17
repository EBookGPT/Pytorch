# Chapter 13: Debugging and Visualization in PyTorch

Welcome back! In our last chapter, we learned about data handling in PyTorch. Now, we move on to an equally important topic: debugging and visualization. As with any programming language, debugging is a crucial aspect of making sure that your PyTorch models run smoothly.

To help us navigate through this chapter, we have a special guest: Adam Paszke. Adam is one of the core developers of PyTorch and has made significant contributions to its development. He is also a co-founder of the AI research lab OpenAI.

In this chapter, we'll cover the following topics:

1. Debugging techniques in PyTorch
2. Visualizing PyTorch models using Tensorboard
3. Use of PyTorch debugger

Debugging is an essential part of the data science process, and PyTorch offers several tools to help you debug your code. We'll discuss some of these tools, including the PyTorch debugger, which can be used to track down errors in your code quickly.

We'll also take a look at a powerful visualization tool called Tensorboard. Tensorboard is a web-based tool that allows you to visualize your machine learning model's training progress and identify potential problems in the model.

Throughout this chapter, we'll use code examples to demonstrate how these debugging and visualization techniques work in practice. We'll also refer to published journals for further reading. 

So, buckle up, and let's dive into the world of debugging and visualization in PyTorch, with our special guest, Adam Paszke!
# The Wizard of Oz and the Debugging Adventure in PyTorch

Once upon a time, there was a young data scientist named Dorothy who lived in the land of PyTorch. She dreamed of building the perfect machine learning model and winning the grand AI competition. So, she worked hard day and night, crafting complex models, tweaking hyperparameters, and training on massive datasets.

However, one day, her code broke, and her model suddenly stopped training. She had no idea what went wrong, and her frustration grew more significant as the hours passed. Eventually, she decided to ask for help from the PyTorch master, the Wizard of Oz.

She followed the yellow brick road, and after a long journey, finally reached the Wizard's castle. As she approached, the door creaked open, and Dorothy stepped into the great hall.

There she saw the Wizard, surrounded by his PyTorch apprentices. She hesitated at first but then gathered her courage and approached the wise old man to ask for help.

The Wizard listened carefully to her story and nodded his head in understanding. "Ah, debugging and visualization. These are essential skills that every data scientist must learn," he said. "Luckily, I have just the person to teach you: Adam Paszke."

Dorothy was amazed to meet one of the core developers of PyTorch and was eager to learn from him. Adam began teaching her about the importance of debugging code, looking for errors and inconsistencies. He showed her how to use the PyTorch debugger, a tool that can detect when the code is not running as expected.

Adam then walked Dorothy through the process of visualization in PyTorch, teaching her how to use Tensorboard to monitor her model's training process, identify potential issues, and make adjustments to get better results.

As Dorothy listened and learned, she realized how powerful these tools were and how they could help transform her work. She thanked Adam and promised to use the techniques he taught her in her future projects.

She left the castle, ready to resume her journey, filled with newfound knowledge, and thanking the Wizard for his guidance. From then on, she knew that she would always be able to find her way out of any debugging rabbit holes or visualize her models with clarity, just like Dorothy did in the land of PyTorch.

And so, just like Dorothy, you too can become a master of PyTorch debugging and visualization, thanks to our special guest, Adam Paszke, and the PyTorch community's extensive knowledge and expertise.
Sure, let's dive into the PyTorch code used to resolve the Wizard of Oz parable!

In this chapter, we learned about two critical components of debugging and visualization in PyTorch: PyTorch Debugger and Tensorboard.

## PyTorch Debugger

PyTorch Debugger is a tool that can help you debug your code by providing detailed information on error messages, breakpoints, and code execution flow.

Here's an example of how to use PyTorch Debugger:

```
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
debugger = torch_debugger.Debugger()
output = debugger.run(model(x))
```

In the code above, we first define our neural network model, which is a simple convolutional neural network. Then, we create an instance of the `torch_debugger.Debugger()` class.

Finally, we use the `run()` method of the debugger to execute our model's forward pass, and PyTorch Debugger provides us with detailed information on the execution flow, errors, memory usage, etc.

## Tensorboard

Tensorboard is a visualization tool that can help you monitor your model's training progress, visualize your model graph, track your model's performance, and identify potential issues. Here's an example of how to use Tensorboard with PyTorch:

```
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
...
for step, (inputs, labels) in enumerate(trainloader):
   # forward + backward + optimize
   outputs = net(inputs)
   loss = criterion(outputs, labels)
   loss.backward()
   optimizer.step()

   # print statistics
   running_loss += loss.item()
   if step % 2000 == 1999:    # print every 2000 mini-batches
       writer.add_scalar('training_loss',
                   running_loss / 2000,
                   epoch * len(trainloader) + i)
       running_loss = 0.0   

# close tensorboard writer
writer.close()
```

In this example, we first import the `SummaryWriter` class from the `torch.utils.tensorboard` module. Then, we create an instance of the `SummaryWriter` class.

During the training loop, we use the `add_scalar()` method of the writer to add the training loss scalar value at each step, which can then be visualized using Tensorboard.

Finally, we close the Tensorboard writer.

And that's how we resolved the Wizard of Oz parable by using PyTorch Debugger and Tensorboard to debug and visualize our models!


[Next Chapter](14_Chapter14.md)