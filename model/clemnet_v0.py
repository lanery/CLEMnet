##### Libaries #####

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import datetime
import random
from sklearn.model_selection import train_test_split

import augmentations_v3 as aug


##### General variables #####

def variables(training_dir_0, EM_tile_size_0, FM_tile_size_0, EMFM_lvl_d, augmen_0):
    
    global training_dir
    global EM_tile_size
    global FM_tile_size
    global lvl_diff
    global augmen
    
    training_dir = training_dir_0
    EM_tile_size = EM_tile_size_0
    FM_tile_size = FM_tile_size_0   
    lvl_diff = 2**EMFM_lvl_d   
    augmen = augmen_0    
        
    print('General variables loaded')

    
##### Augmentation chances #####

p_hf = 0.5     #horizontal flip
p_vf = 0.5     #vertical flip
p_rot = 1     #n90 degrees rotation
p_elas = 0      #elastic deformation
p_noise = 0     #noise in EM
p_noise_FM = 0     #noise in FM


##### Section Class #####

def getSectionList(section_subfolders, training_dir):
    sections=[]
    for i, section in enumerate(section_subfolders):
        sections.append(Section(training_dir, section, i))
    print('Patches located and all functions related to patches initialized')
    return sections

class Section:
    'combines all functions concerning EM and FM image fetching'
    def __init__(self, training_dir, section_subfolders, count):
        'initialize'
        self.count = count      #number to keep track of different directories
        self.EM_dir = training_dir + section_subfolders["EM"]      #directory of EM section
        self.FM_dir = training_dir + section_subfolders["FM"]      #directory of FM section
        
        self.sectionXYList_tot = self.getXYList()      #list of xy coordinates of EM section
        #self.sectionXYList = self.getROI()            #attempt of finding ROI in FM which is not empty
        self.sectionXYList = self.sectionXYList_tot
        
        self.max_tiles = self.maxTiles()    #maximum number of tiles in both dimensions
        
    def xyFromFilename(self, f):
        'returns x and y coordinates of sections in f'
        section_parts = f.split("_")
        return (int(section_parts[0]), int(section_parts[1]), self.count)
   
    def getXYList(self):
        'returns x and y coordinates of EM tiles' 
        return [Section.xyFromFilename(self, f) for f in os.listdir(self.EM_dir) if f.endswith("0.png")]
    
    def getROI(self):
        'returns roi coordinates'
        zoom_lvl = 2
        deg = 2**(zoom_lvl+2)
        tile_name = "{}_{}_{}.png".format(0,0,zoom_lvl)
        filename = self.FM_dir + tile_name
        im_tile = cv2.imread(filename)
        im_tile = im_tile[:,:,1]      
        
        ## test for first non_zero lines
        for i in range(len(im_tile[:,0])):
            if len(np.nonzero(im_tile[:,i])[0]) != 0:
                x1 = i
                break
        for i in range(len(im_tile[:,0])-1,-1,-1):
            if len(np.nonzero(im_tile[:,i])[0]) != 0:
                x2 = i
                break
        for i in range(len(im_tile[0,:])):
            if len(np.nonzero(im_tile[i,:])[0]) != 0:
                y1 = i
                break
        for i in range(len(im_tile[0,:])-1,-1,-1):
            if len(np.nonzero(im_tile[i,:])[0]) != 0:
                y2 = i
                break
        upp_left = int((y1+2*deg-y1%deg)/deg)+1, int((x1+2*deg-x1%deg)/deg)+1
        down_right = int((y2-y2%deg)/deg)-1, int((x2-x2%deg)/deg)-1
        print(upp_left)
        print(down_right)
        return [section for section in self.sectionXYList_tot if upp_left[0] < section[0] <= down_right[0] and upp_left[1] < section[1] <= down_right[1]]
    
    def maxTiles(self):
        'returns max # of tiles in width and height'
        max_i = max(self.sectionXYList,key=lambda item:item[0])[0]
        max_j = max(self.sectionXYList,key=lambda item:item[1])[1]
        return np.array([max_i,max_j]) 
      
    def getTile(self, tile_xy, EM_case):
        'read files'
        if EM_case == True:
            tile_name = "{}_{}_{}.png".format(tile_xy[0], tile_xy[1], 0)
            filename = self.EM_dir + tile_name
            im_tile = cv2.imread(filename)
            if isinstance(None, type(im_tile)):
                im_tile = np.zeros((EM_tile_size,EM_tile_size,1))
                im_tile = im_tile[:, :, 0]
            else: 
                im_tile = im_tile[:, :, 1]  # use the first channel (second and third are the same)
        else:
            tile_name = "{}_{}_{}.png".format(tile_xy[0], tile_xy[1], 5)
            filename = self.FM_dir + tile_name
            im_tile = cv2.imread(filename)
            if isinstance(None, type(im_tile)):
                im_tile = np.zeros((EM_tile_size,EM_tile_size,1))
                im_tile = im_tile[:, :, 0]
            else: 
                im_tile = im_tile[:, :, 1]  # use the first channel (second and third are the same)
        return im_tile
    
    def createPatch(self, tile_size, tile_xy, patch_size, coords, EM_case):
        'Returns patch'
        if (coords[0] + patch_size) > tile_size and (coords[1] + patch_size) > tile_size:
            #get all adjacent tiles
            im_tile_0 = self.getTile((tile_xy[0], tile_xy[1]), EM_case)
            im_tile_1 = self.getTile((tile_xy[0]+1, tile_xy[1]), EM_case)
            im_tile_2 = self.getTile((tile_xy[0], tile_xy[1]+1), EM_case)
            im_tile_3 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
            #new coordinates in adjacent tiles
            coords2 = (coords[0] + patch_size) % tile_size, (coords[1] + patch_size) % tile_size
            #concatenate patches
            im_patch_0 = np.concatenate((im_tile_0[coords[0]:, coords[1]:],
                                             im_tile_1[:coords2[0], coords[1]:]), axis=0)
            im_patch_1 = np.concatenate((im_tile_2[coords[0]:, :coords2[1]],
                                                im_tile_3[:coords2[0], :coords2[1]]), axis=0)
            im_patch = np.concatenate((im_patch_0,
                                                im_patch_1), axis=1)
        elif (coords[0] + patch_size) > tile_size:
            #get bottom adjacent tile
            im_tile_0 = self.getTile((tile_xy[0], tile_xy[1]), EM_case)
            im_tile_1 = self.getTile((tile_xy[0] + 1, tile_xy[1]), EM_case)
            #new coordinates in adjacent tile
            coords_2 = (coords[0] + patch_size) % tile_size
            #concatenate patches
            im_patch = np.concatenate((im_tile_0[coords[0]:, coords[1]:(coords[1]+patch_size)],
                                                im_tile_1[:coords_2, coords[1]:(coords[1]+patch_size)]), axis=0)
        elif (coords[1] + patch_size) > tile_size:
            #get right adjacent tile
            im_tile_0 = self.getTile((tile_xy[0], tile_xy[1]), EM_case)
            im_tile_2 = self.getTile((tile_xy[0], tile_xy[1] + 1), EM_case)
            #new coordinates in adjacent tile
            coords_2 = (coords[1] + patch_size) % tile_size
            #concatenate patches
            im_patch = np.concatenate((im_tile_0[coords[0]:(coords[0]+patch_size), coords[1]:],
                                        im_tile_2[coords[0]:(coords[0]+patch_size), :coords_2]), axis=1)
        else:
            #get tile
            im_tile_0 = self.getTile((tile_xy[0], tile_xy[1]), EM_case)
            #create patch
            im_patch = im_tile_0[coords[0] : (coords[0]+patch_size),coords[1]:(coords[1]+patch_size)]
        return im_patch
    
    def createPatch_test(self, tile_size, tile_xy, patch_size, coords, edge_pixels, EM_case):
        'Returns patch for testing, edge_pixels larger than required size of patch'
        im_tile_1 = self.getTile((tile_xy[0]-1, tile_xy[1]-1), EM_case)
        im_tile_2 = self.getTile((tile_xy[0]-1, tile_xy[1]), EM_case)
        im_tile_3 = self.getTile((tile_xy[0]-1, tile_xy[1]+1), EM_case)
        im_tile_4 = self.getTile((tile_xy[0], tile_xy[1]-1), EM_case)
        im_tile_5 = self.getTile((tile_xy[0], tile_xy[1]), EM_case)
        im_tile_6 = self.getTile((tile_xy[0], tile_xy[1]+1), EM_case)
        im_tile_7 = self.getTile((tile_xy[0]+1, tile_xy[1]-1), EM_case)
        im_tile_8 = self.getTile((tile_xy[0]+1, tile_xy[1]), EM_case)
        im_tile_9 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
        
        em_edge = edge_pixels * lvl_diff
        
        im_patch_0 = np.concatenate((im_tile_1[-em_edge:,-em_edge:], im_tile_2[-em_edge:,:], 
                                     im_tile_3[-em_edge:,:em_edge]), axis=1)
        im_patch_1 = np.concatenate((im_tile_4[:,-em_edge:], im_tile_5, im_tile_6[:,:em_edge]), axis=1)
        im_patch_2 = np.concatenate((im_tile_7[:em_edge,-em_edge:], im_tile_8[:em_edge,:],
                                     im_tile_9[:em_edge,:em_edge]), axis=1)
        
        im_patch = np.concatenate((im_patch_0, im_patch_1, im_patch_2), axis=0)
    
        return im_patch
    
    def getPatch(self, tile_xy, EM_patch_size, FM_patch_size):
        'returns patch of EM and corresponding FM image with random'
        #get (random) coordinates
        EM_coords = randCoord()
        FM_coords_setup = ( (tile_xy[0] % lvl_diff)*FM_patch_size, (tile_xy[1] % lvl_diff)*FM_patch_size)
        FM_coords = int(FM_coords_setup[0]+EM_coords[0]/FM_patch_size), int(FM_coords_setup[1]+EM_coords[1]/FM_patch_size)
        #get corresponding FM tile
        FM_tile_xy = (tile_xy[0] // lvl_diff, tile_xy[1] // lvl_diff)
        #get patch
        im_EM_patch = self.createPatch(EM_tile_size, tile_xy, EM_patch_size, EM_coords, EM_case = True)
        im_FM_patch = self.createPatch(FM_tile_size, FM_tile_xy, FM_patch_size, FM_coords, EM_case = False)
        return im_EM_patch, im_FM_patch
            
    def getPatchNonRandom(self, tile_xy, EM_patch_size, FM_patch_size):
        'returns patch of EM and corresponding FM image'
        #get standard coordinates
        EM_coords = 0, 0
        FM_coords_setup = ( (tile_xy[0] % lvl_diff)*FM_patch_size, (tile_xy[1] % lvl_diff)*FM_patch_size)
        FM_coords = int(FM_coords_setup[0]+EM_coords[0]/FM_patch_size), int(FM_coords_setup[1]+EM_coords[1]/FM_patch_size)
        #get corresponding FM tile
        FM_tile_xy = (tile_xy[0] // lvl_diff, tile_xy[1] // lvl_diff)
        #get patch
        im_EM_patch = self.createPatch(EM_tile_size, tile_xy, EM_patch_size, EM_coords, EM_case = True)
        im_FM_patch = self.createPatch(FM_tile_size, FM_tile_xy, FM_patch_size, FM_coords, EM_case = False)
        return im_EM_patch, im_FM_patch
        
    def getPatchTest_old(self, tile_xy, EM_patch_size, EM_coords):
        'returns patch of EM'
        #get patch
        im_EM_patch = self.createPatch(EM_tile_size, tile_xy, EM_patch_size, EM_coords, EM_case = True)
        return im_EM_patch
    
    def getPatchTest(self, tile_xy, EM_patch_size, EM_coords, edge_pixels):
        'returns patch of EM'
        #get patch
        im_EM_patch = self.createPatch_test(EM_tile_size, tile_xy, EM_patch_size, EM_coords, edge_pixels, EM_case = True)
        return im_EM_patch
    
    
    
##### Randomizers #####
def randCoord():
    'Gives kind of random coordinates'
    coord_sizes = []
    for i in range(0, int(EM_tile_size/lvl_diff), 2):
        coord_sizes.append(i * lvl_diff)
    coords = coord_sizes[random.randint(0, len(coord_sizes)-1)], coord_sizes[random.randint(0, len(coord_sizes)-1)]
    return coords

def randPatch_size():
    'returns random EM and corresponding EM patch sizes'
    n_pool_layers = 6     #defined by model archictecture
    min_patch_size = 18 + n_pool_layers     #defined by model archictecture or/and architect
    max_patch_size = 1024 / lvl_diff    #limited by (GPU) memory
    patch_sizes = []
    for i in range(min_patch_size, int(max_patch_size+1), 2):
        patch_sizes.append(i * lvl_diff)
    print(patch_sizes)
    EM_patch_size = patch_sizes[random.randint(0, len(patch_sizes)-1)]
    FM_patch_size = EM_patch_size // lvl_diff   
    return EM_patch_size, FM_patch_size

def randTile(max_tile):
    'returns random tile coordinates'
    #Get random tile, excluding right and bottom border tiles
    tile_xy = random.randint(0,max_tile[0]-1), random.randint(0,max_tile[1]-1)
    return tile_xy




##### Training Data Generator with randomizers#####
class DataGenerator(keras.utils.Sequence):
    #### What to do with validation? One total training dataset use for that? 
    def __init__(self, sections, batch_size, n_folder):
        self.sections = sections      #contains all info over sections
        self.batch_size = batch_size      #batch size
                
        self.x_set = []
        self.folders = []
        for i in range(n_folder):
            self.x_set.extend(np.array(self.sections[i].sectionXYList))
            self.folders.append(i)
        self.folders = np.array(self.folders)
        self.x_set = np.array(self.x_set)
        
        self.on_epoch_end()     #to let keras know there is on_epoch_end function

    def __len__(self):
        'returns the amount of batches per epoch'
        return int(np.floor(len(self.x_set)/self.batch_size))

    def __getitem__(self, idx):
        'returns a batch'           
        #Get random patch sizes
        EM_patch_size, FM_patch_size = randPatch_size()
        
        batch_x = np.empty((self.batch_size, EM_patch_size, EM_patch_size, 1))     #initialise EM images
        batch_y = np.empty((self.batch_size, FM_patch_size, FM_patch_size, 1))     #initialise FM image
                
        for i in range(0, self.batch_size):
            'fetch all images for current batch'
            #Get random folder
            randFolder = self.folders[random.randint(0,len(self.folders)-1)]
            
            #Get random tile, excluding right and bottom border tiles
            tile_xy = randTile(self.sections[randFolder].n_tiles)
            
            #Get one EM and corresponding FM image
            im_EM_patch, im_FM_patch = self.sections[randFolder].getPatch(tile_xy, EM_patch_size, FM_patch_size)
            
            if augmen == True:
                #Transformations
                im_EM_patch, im_FM_patch = aug.hor_flip(im_EM_patch, im_FM_patch, p_hf)
                im_EM_patch, im_FM_patch = aug.ver_flip(im_EM_patch, im_FM_patch, p_vf)
                im_EM_patch, im_FM_patch = aug.rot_aug(im_EM_patch, im_FM_patch, p_rot)
                #im_EM_patch, im_FM_patch = aug.elas_aug(im_EM_patch, im_FM_patch, p_elas)
                #im_EM_patch = aug.noise_aug(im_EM_patch, p_noise)
                #im_FM_patch = aug.noise_FM_aug(im_FM_patch, p_noise_FM)
            
            batch_x[i, :, :, 0] = im_EM_patch / 255.
            batch_y[i, :, :, 0] = im_FM_patch / 255.
            
        return batch_x, batch_y   
        
    def on_epoch_end(self):
        'returns loss and accuracy'
        #can add some callbacks
        
        


##### Training Data Generator without randomizers #####
class DataGenerator_NonRand(keras.utils.Sequence):
    def __init__(self, sections, batch_size, val_size, rand_state, n_folder, training):
        self.sections = sections      #contains all info over sections
        self.batch_size = batch_size      #batch size
        
        self.x_set = []         #create list of all available patches
        for i in range(n_folder):
            self.x_set.extend(np.array(self.sections[i].sectionXYList))
        self.x_set = np.array(self.x_set)
        
        self.x_train, self.x_val = train_test_split(self.x_set, test_size=val_size, random_state=rand_state)     #divide all patches into train and validation set
        
        if training == True and len(self.x_train) == 1:      #if only training on one patch
            print('Model only training on patch: ', self.x_train[0])
        
        self.training = training      #set training to either training or validating
        self.seed = 1                 #seed to 'randomize' order of patches every epoch
        self.on_epoch_end()     #to let keras know there is on_epoch_end function
        
    def __len__(self):
        'returns the amount of batches per epoch'
        if self.training == True:
            epoch_length = int(np.floor(len(self.x_train)/self.batch_size))
        else:
            epoch_length = int(np.floor(len(self.x_val)/self.batch_size))
        return epoch_length

    def __getitem__(self, idx):
        'returns a batch'           
        EM_patch_size, FM_patch_size = 1024, 32     #get standard patch sizes
        
        batch_x = np.empty((self.batch_size, EM_patch_size, EM_patch_size, 1))     #initialise EM images
        batch_y = np.empty((self.batch_size, FM_patch_size, FM_patch_size, 1))     #initialise FM image
        
        if self.training == True:     #training phase
            for i, samp in enumerate(self.x_train[idx*self.batch_size:(idx+1)* self.batch_size]):
                'fetch all images for current batch'
                #Get tile coordinates
                tile_xy = samp[0], samp[1]
                #Get one EM and corresponding FM image
                im_EM_patch, im_FM_patch = self.sections[samp[2]].getPatchNonRandom(tile_xy, EM_patch_size, FM_patch_size)
                
                if augmen == True:
                    #Transformations
                    im_EM_patch, im_FM_patch = aug.hor_flip(im_EM_patch, im_FM_patch, p_hf)
                    im_EM_patch, im_FM_patch = aug.ver_flip(im_EM_patch, im_FM_patch, p_vf)
                    im_EM_patch, im_FM_patch = aug.rot_aug(im_EM_patch, im_FM_patch, p_rot)

                #Make sure values are between 0 and 1 for binary loss function
                batch_x[i, :, :, 0] = im_EM_patch / 255.
                batch_y[i, :, :, 0] = im_FM_patch / 255.
        else:        #validation phase
            for i, samp in enumerate(self.x_val[idx*self.batch_size:(idx+1)* self.batch_size]):
                'fetch all images for current batch'
                #Get tile coordinates
                tile_xy = samp
                #Get one EM and corresponding FM image
                im_EM_patch, im_FM_patch = self.sections[samp[2]].getPatchNonRandom(tile_xy, EM_patch_size, FM_patch_size)

                batch_x[i, :, :, 0] = im_EM_patch / 255.
                batch_y[i, :, :, 0] = im_FM_patch / 255.
        
        return batch_x, batch_y
        
    def on_epoch_end(self):
        'randomize order of patches'
        perm_1 = np.random.RandomState(seed=self.seed).permutation(len(self.x_train))
        self.x_train = self.x_train[perm_1]
        perm_2 = np.random.RandomState(seed=self.seed).permutation(len(self.x_val))
        self.x_val = self.x_val[perm_2] #not necessary
        self.seed += 1

        
##### Testing Data Generator #####
class TestGenerator(keras.utils.Sequence):
    def __init__(self, sections, batch_size, EM_coords, edge_pixels, evaluation):
        self.sections = sections[0]      #contains all info over sections
        self.batch_size = batch_size     #batch size
                
        self.x_set = np.array(self.sections.sectionXYList)   #list of all patches
        
        self.EM_dir = self.sections.EM_dir     #EM directory
        
        self.EM_coords = EM_coords      #coordinates in EM patch (pixels)
        
        self.edge_pixels = edge_pixels
        self.evaluation = evaluation

    def __len__(self):
        'returns the amount of batches per epoch'
        return int(np.floor(len(self.x_set)/self.batch_size))

    def __getitem__(self, idx):
        'returns a batch'     
        
        if self.evaluation:
            EM_patch_size, FM_patch_size = EM_tile_size, int(EM_tile_size/lvl_diff)     #get standard patch sizes 
            batch_x = np.empty((self.batch_size, EM_patch_size, EM_patch_size, 1))     #initialise EM images
            batch_y = np.empty((self.batch_size, FM_patch_size, FM_patch_size, 1))     #initialise FM image
            for i, samp in enumerate(self.x_set[idx*self.batch_size : (idx+1)*self.batch_size]):
                'fetch all images for current batch'
                #get tile coordinates
                tile_xy = samp

                #Get one EM and corresponding FM image
                im_EM_patch, im_FM_patch = self.sections.getPatchNonRandom(tile_xy, EM_patch_size, FM_patch_size)

                batch_x[i, :, :, 0] = im_EM_patch / 255.
                batch_y[i, :, :, 0] = im_FM_patch / 255.
                
            return batch_x, batch_y
        
        else:
            
            #Get patch sizes same as EM tile size 
            EM_patch_size = EM_tile_size + 32*self.edge_pixels*2
            FM_patch_size = int(EM_tile_size/lvl_diff) 

            batch_x = np.empty((self.batch_size, EM_patch_size, EM_patch_size, 1))     #initialise EM images
            batch_y = np.empty((self.batch_size, FM_patch_size, FM_patch_size, 1))     #initialise FM image

            for i, samp in enumerate(self.x_set[idx*self.batch_size : (idx+1)*self.batch_size]):
                'fetch all images for current batch'
                #get tile coordinates
                tile_xy = samp

                #Get one EM and corresponding FM image
                im_EM_patch = self.sections.getPatchTest(tile_xy, EM_patch_size, self.EM_coords, self.edge_pixels)
                im_EM_patch_2, im_FM_patch = self.sections.getPatchNonRandom(tile_xy, EM_patch_size, FM_patch_size)

                batch_x[i, :, :, 0] = im_EM_patch / 255.
                batch_y[i, :, :, 0] = im_FM_patch / 255.

            return batch_x, batch_y       
        

##### Testing Data Generator One value#####
class TestGeneratorOneValue(keras.utils.Sequence):
    def __init__(self, sections, batch_size, rand_state):
        self.sections = sections[0]      #contains all info over sections
        self.batch_size = batch_size      #batch size
                
        self.x_set = np.array(self.sections.sectionXYList)
        self.x_train, self.x_val = train_test_split(self.x_set, test_size=0.999, random_state=rand_state)
        print('Model only training on patch: ', self.x_train[0])
        
        self.EM_dir = self.sections.EM_dir
        self.FM_dir = self.sections.FM_dir
        self.n_tiles = self.sections.n_tiles
    
    def patch_search(self):
        'returns patch'
        return self.x_train[0]

    def __len__(self):
        'returns the amount of batches per epoch'
        return int(np.floor(len(self.x_train)/self.batch_size))
    
    def __getitem__(self, idx):
        'returns a batch'
        write_output = [0]
        if idx in write_output:
            print('Currently predicting image ', self.x_train[0])
        
        #Get patch sizes same as EM tile size 
        EM_patch_size = EM_tile_size 
        FM_patch_size = EM_patch_size // lvl_diff
        
        batch_x = np.empty((1, EM_patch_size, EM_patch_size, 1))     #initialise EM images
        
        for i, samp in enumerate(self.x_set[idx*self.batch_size : (idx+1)*self.batch_size]):
            'fetch all images for current batch'
            #get tile coordinates
            tile_xy = samp
            
            #Get one EM and corresponding FM image
            im_EM_patch = self.slides.getPatchTest_old(tile_xy, EM_patch_size, self.EM_coords)
            
            batch_x[i, :, :, 0] = im_EM_patch / 255.
            
        return batch_x
    
    
    
     
##### Model #####

def getModel():
    'returns the architecture of the neural network'
    
    img_shape=(None, None, 1)     #input shape of images
    #Note: change 1024 to 'None' to have variable input

    #create U-net architecture
    inputs = keras.layers.Input(shape = img_shape)     #input layer
    
    conv0 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool0 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool0)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    up6 = keras.layers.UpSampling2D(size=(2, 2))(conv6)

    merge7 = keras.layers.concatenate([conv5, up6], axis=3)
    conv7 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    
    conv10 = keras.layers.Conv2D(1, 1, activation= 'sigmoid')(conv7)
    
    model = keras.Model(inputs=inputs, outputs=conv10)     #put model into one variable
    #model.summary()     #gives a summary of model in notebook
    
    return model 

##### Model for prediction
def getModel_pred():
    'returns the architecture of the neural network'
    
    img_shape=(None, None, 1)     #input shape of images

    #create U-net architecture
    inputs = keras.layers.Input(shape = img_shape)     #input layer
    
    conv0 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool0 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool0)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    up6 = keras.layers.UpSampling2D(size=(2, 2))(conv6)

    merge7 = keras.layers.concatenate([conv5, up6], axis=3)
    conv7 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    
    conv10 = keras.layers.Conv2D(1, 1, activation= 'sigmoid')(conv7)
    
    layer11 = keras.layers.Cropping2D(cropping=((4,4),(4,4)))(conv10)
    
    model = keras.Model(inputs=inputs, outputs=layer11)     #put model into one variable
    #model.summary()     #gives a summary of model in notebook
    
    return model 
