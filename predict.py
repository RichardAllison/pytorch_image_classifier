import argparse
import json
import numpy as np
import torch
import checkpoint_helper as ch
import image_processor as ip
from PIL import Image
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session

def main():
    #Get input arguments
    args = get_input_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    gpu = args.gpu
    category_names = args.category_names
            
    device = torch.device("cuda" if args.gpu else "cpu")    
    model = ch.load_checkpoint(checkpoint)[0]
    model.to(device)
    
    top_ps, top_classes = predict(image_path, model, device, top_k)
    ps = top_ps.cpu().data.numpy().squeeze()
    
    cat_to_name = get_cat_to_name(category_names)
    labels = get_labels(cat_to_name, top_classes)

    display_results(ps, labels)
    

def display_results(top_ps, labels):
    '''
    Print the top result(s) in the console
    '''
    if len(labels) == 1:
        print("Top prediction:")
        print("{}: {:.2f}%".format(labels[0], top_ps * 100))
    else: 
        print("Top predictions:")
        for top_p, label in zip(top_ps, labels):
            print("{}: {:.2f}%".format(label, top_p * 100))
    
def get_cat_to_name(filepath):
    '''Get the category names from file passed in
    '''
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def get_labels(cat_to_name, top_classes):
    '''
    Get the labels for the top classes from the category names passed in
    '''
    labels = []    
    for index in top_classes:
        labels.append(cat_to_name[str(index)])
    labels = np.array(labels)
    return labels


def predict(image_path, model, device, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device)

    img = Image.open(image_path)
    img = ip.process_image(image_path)
    img = torch.from_numpy(img)
    img = img.unsqueeze_(0)
    if device.type == "cuda":
        img = img.type(torch.cuda.FloatTensor)
    else:
        img = img.type(torch.FloatTensor)
    img.to(device)
        
    class_to_idx = dict([(value, key) for key, value in model.class_to_idx.items()])

    with torch.no_grad():
        
        logps = model.forward(img)
        ps = torch.exp(logps) 
        top_ps, top_classes = ps.topk(topk, dim=1)

        indexes = []        
        for index in top_classes.cpu().data.numpy()[0]:
            indexes.append(class_to_idx.get(index))
        indexes = np.array(indexes)
            
    return top_ps, indexes


def get_input_args():
    '''
    Get input arguments passed in from the command line
    '''
    parser = argparse.ArgumentParser(description="Get Input Arguments")
    
    parser.add_argument("image_path", type=str, action="store", help="Path of image")
    parser.add_argument("checkpoint", type=str, action="store", help="Path to saved checkpoint of trained model")
    parser.add_argument("--top_k", type=int, default="1", help="The number of the highest probabilities")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path of category names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
     
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()