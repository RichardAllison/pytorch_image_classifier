import torch
import classifier as cf

def save_checkpoint(model, pretrained_model, classifier, optimiser, class_to_idx, directory):
    '''
    Save checkpoint to the directory specified with the parameters passed in 
    '''
    filepath = "{}/checkpoint.pth".format(directory)
    checkpoint = {
             'pretrained_model': pretrained_model,
             'input_size': classifier.input_size,
             'output_size': classifier.output_size,
             'hidden_layers': [layer.out_features for layer in classifier.hidden_layers],
             'dropout': classifier.dropout.p,
             'learn_rate': classifier.learn_rate,
             'epochs': classifier.epochs,
             'class_to_idx': class_to_idx,
             'state_dict': model.state_dict(),
             'optimiser': optimiser,
             'optimiser_state_dict': optimiser.state_dict()
    }
    torch.save(checkpoint, filepath)
    
    
def load_checkpoint(filepath): 
    '''
    Load checkpoint spefified in the filepath passed in
    '''
    checkpoint = torch.load(filepath)
    model = checkpoint['pretrained_model']
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = cf.Classifier(checkpoint['input_size'],
                       checkpoint['hidden_layers'],
                       checkpoint['output_size'],
                       checkpoint['dropout'],
                       checkpoint['learn_rate'],
                       checkpoint['epochs'])
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimiser = checkpoint['optimiser']
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    return model, optimiser