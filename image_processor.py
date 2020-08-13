import numpy as np
from PIL import Image

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array
    '''
    pil_image = Image.open(image)
    
    resize_dimensions = (256,256)
    crop_dimensions = (224,224)

    #Resize image
    pil_image = resize_image(pil_image, resize_dimensions)
    
    #Crop center 224 x 224 square of image
    pil_image = crop_image(pil_image, crop_dimensions)
    
    #Convert image to numpy array
    np_image = np.array(pil_image)
    
    #Convert RGB values to between 0-1
    np_image = np_image/255

    #Normalise the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    #Transpose columns
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def resize_image(pil_image, dimensions):
    ''' 
    Resize shortest side to new dimension and keep aspect ratio
    Returns PIL Image
    '''
    width, height = pil_image.size
    if (width > height):
        new_height = dimensions[1]
        new_width = round(width * (new_height/height))
    else:
        new_width = dimensions[0]
        new_height = round(height * (new_width/width))     

    return pil_image.resize((new_width, new_height))


def crop_image(pil_image, dimensions):
    ''' 
    Crop the center of a pil_image to the new dimensions
    Returns PIL Image
    '''
    horizontal_center = pil_image.width/2
    vertical_center = pil_image.height/2
    
    crop_left = horizontal_center - dimensions[0]/2
    crop_top = vertical_center - dimensions[1]/2
    crop_right = horizontal_center + dimensions[0]/2
    crop_bottom = vertical_center + dimensions[1]/2
   
    return pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))