##### libaries #####
import numpy as np
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


##### augmentation functions #####

def hor_flip(im_EM, im_FM, p_hf):
    'Horizontal flip'
    if random.random() <=p_hf:
        im_EM = np.flip(im_EM, 1)
        im_FM = np.flip(im_FM, 1)
        
    return im_EM, im_FM

def ver_flip(im_EM, im_FM, p_vf):
    'Vertical flip'
    if random.random() <=p_vf:
        im_EM = np.flip(im_EM, 0)
        im_FM = np.flip(im_FM, 0)
        
    return im_EM, im_FM

def rot_aug(im_EM, im_FM, p_rot):
    'Random n times 90 degrees rotation'
    p_total = random.random()
    if p_total > 0.75:
        im_EM = np.rot90(im_EM, 1, axes=(1,0))
        im_FM = np.rot90(im_FM, 1, axes=(1,0))
    elif p_total > 0.5:
        im_EM = np.rot90(im_EM, 2, axes=(1,0))
        im_FM = np.rot90(im_FM, 2, axes=(1,0))
    elif p_total > 0.25:
        im_EM = np.rot90(im_EM, 3, axes=(1,0))
        im_FM = np.rot90(im_FM, 3, axes=(1,0))
    
    return im_EM, im_FM

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Slightly modified to work for 2D images, 2019 (Balkenende)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y= np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def elas_aug(im_EM, im_FM, p_elas):
    'Elastic deformation'
    #Be aware that this can change the FM image in a different way.
    #However, does the FM image needs to be deformed??
    # I dont think so, maybe instead of elastic deformation, make function for astigmatism
    
    alpha = 1000  #scaling of the gaussian
    sigma = 40   #standard deviation of gaussian kernel
    
    if random.random() <=p_elas:
        im_EM = elastic_transform(im_EM, alpha, sigma)
        
    return im_EM, im_FM

def noise_aug(im_EM, p_noise):
    'Noise for EM images'
    if random.random() <=p_noise:
        noise = 1  #amount of noise
        noise = random.random()  #amount of noise
        im_EM = im_EM + noise * im_EM.std() * np.random.random(im_EM.shape)
        
    return im_EM

def noise_FM_aug(im_EM, p_noise):
    'Noise for FM images'
    #Deze moet nog worden bijgewerkt
    if random.random() <=p_noise:
        noise = 1  #amount of noise
        noise = random.random()  #amount of noise
        im_EM = im_EM + noise * im_EM.std() * np.random.random(im_EM.shape)
        
    return im_EM


    
    