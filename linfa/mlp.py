import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    """Fully Connected Neural Network"""

    def __init__(self, input_size, output_size, arch=None, activation='relu', device='cpu', init_zero=False, dropout=None):
        """
        Args:
            input_size (int): Input size for FNN
            output_size (int): Output size for FNN
            arch (list of int): list containing the number of neurons for each hidden layer
            activation (stirng): type of activation function, either 'relu', 
            device (string): computational device
        """
        super().__init__()
        if(arch is None):
            self.neuron_size = [64,32]
        else:
            self.neuron_size = arch
        # Set activation
        self.activation = activation
        self.fc = nn.ModuleList()
        # Assign first layer
        self.fc.append(nn.Linear(input_size, self.neuron_size[0]).to(device))
        if(len(self.neuron_size) == 1):
            # Assign output Layer
            self.fc.append(nn.Linear(self.neuron_size[0], output_size).to(device))
        else:
            for layer in range(1,len(self.neuron_size)):
                self.fc.append(nn.Linear(self.neuron_size[layer-1], self.neuron_size[layer]).to(device))
            # Assign output layer
            self.fc.append(nn.Linear(self.neuron_size[len(self.neuron_size)-1], output_size).to(device))
        # Add dropout layers
        self.dropout = dropout
        if(self.dropout is not None):
            self.dropout_visible = nn.Dropout(p=self.dropout[0])
            self.dropout_hidden = nn.Dropout(p=self.dropout[1])


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [num_samples, input_size], assumed to be a batch.

        Returns:
            torch.Tensor. Assumed to be a batch.
        """
        if(self.dropout is not None):

            x = self.dropout_visible(x)

        for loopA in range(len(self.fc)-1):

            if(self.activation == 'relu'):                

                if(self.dropout is not None):
                    x = self.dropout_hidden(F.relu(self.fc[loopA](x)))
                else:
                    x = F.relu(self.fc[loopA](x))

            elif(self.activation == 'silu'):

                if(self.dropout is not None):
                    x = self.dropout_hidden(F.silu(self.fc[loopA](x)))
                else:
                    x = F.silu(self.fc[loopA](x))

            elif(self.activation == 'tanh'):

                if(self.dropout is not None):
                    x = self.dropout_hidden(F.tanh(self.fc[loopA](x)))
                else:
                    x = F.tanh(self.fc[loopA](x))

            else:
                print('Invalid activation string.')
                exit(-1)

        # Last layer with linear activation
        x = self.fc[-1](x)
        return x
