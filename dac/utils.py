import numpy as np
import os
from PIL import Image
import torch

def flatten_image(pil_image):
    """
    pil_image: image as returned from PIL Image
    """
    return np.array(pil_image[:,:,0], dtype=np.float32)

def normalize_image(image):
    """
    image: 2D input image
    """
    return (image.astype(np.float32)/255. - 0.5)/0.5

def open_image(image_path, flatten=False, normalize=False):
    im = np.asarray(Image.open(image_path))
    if flatten:
        im = flatten_image(im)
    if normalize:
        im = normalize_image(im)
    return im

def image_to_tensor(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = torch.tensor(image, device=device)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    return image_tensor

def save_image(array, image_path, renorm=True, norm=False):

    if renorm:
        array = (array *0.5 + 0.5)*255
    if norm:
        array/=np.max(np.abs(array))
        array *= 255
    #print("initial save type")
    #print(type(array))
    arrayb = []
    arrayb = array
    #arrayb = torch.from_numpy(array)
    if torch.is_tensor(array):
        #print(array.shape)
        array = array.squeeze()

        #print(array.shape)
        #print("moving to cpu")
        arrayb = array.cpu() #numpy.ndarray has no attribute cpu
        #if len(array.shape) == 3:
        #    arrayb = arrayb.permute(1,2,0)
        arrayb = np.asarray(arrayb) #cannot convert cuda device type tensor to numpy
    #print(type(arrayb))
    #print(image_path)
    #print("output image shape ", str(arrayb.shape)) #masks are 256,256,3. attr, real, fake are all 256,256
    #print(arrayb)
    #print(arrayb.shape)
    #for some reason saving a 2D numpy array "as RGB" generates a rainbow image, not a grayscale image
    if len(arrayb.shape) == 2:
        im = Image.fromarray(arrayb, mode='L')
    else:
        im = Image.fromarray(arrayb.astype(np.uint8),'RGB')
        #pixels = np.asarray(im)
    #print(pixels)
    #im = Image.fromarray(arrayb)
    #im = im.convert('RGB')
    im.save(image_path)

def get_all_pairs(classes):
    pairs = []
    i = 0
    for i in range(len(classes)):
        for k in range(i+1, len(classes)):
            pair = (classes[i], classes[k])
            pairs.append(pair)

    return pairs

def get_image_pairs(base_dir, class_0, class_1):
    """
    Experiment datasets are expected to be placed at 
    <base_dir>/<class_0>_<class_1>
    """
    image_dir = f"{base_dir}/{class_0}_{class_1}"
    images = os.listdir(image_dir)
    real = [os.path.join(image_dir,im) for im in images if "real" in im and im.endswith(".png")]
    fake = [os.path.join(image_dir,im) for im in images if "fake" in im and im.endswith(".png")]
    paired_images = []
    for r in real:
        for f in fake:
            if r.split("/")[-1].split("_")[-1] == f.split("/")[-1].split("_")[-1]:
                paired_images.append((r,f))
                break

    return paired_images
