import argparse
import numpy as np
import torch
import checkpoint_helper as ch
import classifier as cf
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session

def main():
    #Get input arguments
    args = get_input_args()
    data_dir = args.data_dir    
    device = torch.device("cuda" if args.gpu else "cpu")
    arch = args.arch
    hidden_layers = args.hidden_units
    criterion = args.data_dir
    learn_rate = args.learning_rate
    epochs = args.epochs
    save_dir = args.save_dir
    validation_interval = 10
    dropout = 0.02
    
    #Get pretrained model
    pretrained_model = load_pretrained_model(arch)
    model = pretrained_model
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    #Import data
    dataloaders, image_datasets = import_data(data_dir)
    class_to_idx = image_datasets["training"].class_to_idx
    
    #Get input and output parameters
    input_size = get_input_size_from_model(model)
    output_size = len(class_to_idx)
    
    #Define classifier
    classifier = cf.Classifier(input_size, 
                               hidden_layers, 
                               output_size, 
                               dropout, 
                               learn_rate, 
                               epochs)
    model.classifier = classifier
    model.to(device)
    
    #Define criterion
    criterion = nn.NLLLoss()
    
    #Define optimiser
    optimiser = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    #Train model
    train(model, 
          device, 
          dataloaders["trainloader"], 
          dataloaders["validloader"], 
          criterion, 
          optimiser, 
          learn_rate, 
          epochs, 
          validation_interval)

    #Save checkpoint
    ch.save_checkpoint(model,
                       pretrained_model,
                       model.classifier,
                       optimiser,
                       class_to_idx,
                       save_dir)


def import_data(data_dir):
    '''
    Import the data to be used in training, validation and testing.
    Folder must contain 'train', 'valid' and 'test' subfolders.
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {"training": transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ]),
                       "validation": transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                     ]),
                       "testing": transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                     ])
                     }

    image_datasets = {"training": datasets.ImageFolder(
                        train_dir, transform=data_transforms["training"]),
                      "validation": datasets.ImageFolder(
                        valid_dir, transform=data_transforms["validation"]),
                      "testing": datasets.ImageFolder(
                        test_dir, transform=data_transforms["testing"])
                     }

    dataloaders = {"trainloader": torch.utils.data.DataLoader(
                      image_datasets["training"], batch_size=64, shuffle=True),
                   "validloader": torch.utils.data.DataLoader(
                      image_datasets["validation"], batch_size=64, shuffle=True),
                   "testloader": torch.utils.data.DataLoader(
                      image_datasets["testing"], batch_size=64, shuffle=True)
                  }
    
    return dataloaders, image_datasets


def load_pretrained_model(arch):
    '''
    Get a pretrained model from architecture passed in,
    from a choice of alexnet or densenet121
    '''
    if arch == "alexnet":
        return models.alexnet(pretrained=True)
    elif arch == "densenet121":
        return models.densenet121(pretrained=True)


def get_input_size_from_model(model):
    '''
    A method to get the input size of the pretrained model's classifier
    '''
    sizes = []
    for name, param in model.classifier.named_parameters():
        sizes.append(param.size())

    return sizes[0][1]

    
def train(model, device, trainloader, validloader, criterion, 
          optimiser, learn_rate, epochs, validation_interval):
    '''
    Train the model with the parameters passed in
    '''
    with active_session():
        steps = 0
        running_loss = 0
        model.train()
        for epoch in range(epochs):

            for images, labels in trainloader:
                #Transferring images and labels to device
                images, labels = images.to(device), labels.to(device)

                #Set gradient to zero on each loop before performing backpropogation
                optimiser.zero_grad()

                #Feedforward
                output = model.forward(images)

                #Calculating loss from the output
                loss = criterion(output, labels)

                #Performing backpropogation, calculating the gradient of the loss
                loss.backward()

                #Appending the loss to the running loss
                running_loss+=loss.item()

                #Updating the weights, taking a step with the optimiser
                optimiser.step()

                steps += 1

                #Checking accuracy at set intervals 
                if steps % validation_interval == 0:
                    valid_loss = 0
                    accuracy = 0

                    with torch.no_grad():
                        model.eval()
                        for images, labels in validloader: 
                            
                            images, labels = images.to(device), labels.to(device)
                            
                            #Calculating loss
                            logps = model.forward(images)
                            loss = criterion(logps, labels)
                            valid_loss += loss.item()

                            #Calculating accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                        #Printing loss and accuracy to help determine best hyperparameters
                        print("Epoch: {}/{} ".format(epoch+1, epochs),
                              "Training loss: {:.3f}.. ".format(running_loss/validation_interval),
                              "Validation loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                              "Validation accuracy: {:.3f}{}".format(accuracy/len(validloader)*100,"%"))
                        
                        #Resetting running_loss 
                        running_loss = 0
                        
                        #Setting model back to training mode
                        model.train()
        print("Done")
        

def get_input_args():
    '''
    Get input arguments passed in from the command line
    '''
    parser = argparse.ArgumentParser(description="Get Input Arguments")
    parser.add_argument("data_dir", type=str, action="store", help="Path to data directory")
    parser.add_argument("--save_dir", type=str, default=".", help="Save checkpoint location")
    parser.add_argument("--arch", type=str, default="alexnet", choices=["alexnet", "densenet121"], help="Choose CNN model architecture") 
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--hidden_units", nargs='+', type=int, default=[512, 256], help="Hidden units for one or more layers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    
    args = parser.parse_args()
    
    return args

    
if __name__ == "__main__":
    main()