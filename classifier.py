import torch
import torch.nn.functional as F
from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.02, learn_rate=0.003, epochs=10):
        
        super().__init__()
        #Dropout
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout)
        self.learn_rate = learn_rate
        self.epochs = epochs
        
        #Populate hidden layers from input variables
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])        
        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.extend([nn.Linear(layer1, layer2)])
        self.output = nn.Linear(hidden_layers[-1], output_size)    
    
    def forward(self, x):
        #Flatten the image
        x = x.view(x.shape[0], -1)      
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))
            
        #Log softmax output
        x = F.log_softmax(self.output(x), dim=1)
        return x