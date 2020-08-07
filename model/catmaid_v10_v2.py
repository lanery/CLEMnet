##### Libaries #####
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from skimage.color import grey2rgb
from skimage.exposure import rescale_intensity
import ipywidgets as wg


def variables(training_dir_0, EM_tile_size_0, FM_tile_size_0, EMFM_lvl_d):
    
    global training_dir
    global EM_tile_size
    global FM_tile_size
    global lvl_diff
    
    training_dir = training_dir_0
    EM_tile_size = EM_tile_size_0
    FM_tile_size = FM_tile_size_0   
    lvl_diff = 2**EMFM_lvl_d  
        
    print('General variables loaded')

##### Slide Class #####
def getSlideList(slide_subfolders, training_dir, zoom_lvl):
    slides=[]
    for i, slide in enumerate(slide_subfolders):
        #print(i)
        slides.append(Slide(training_dir, slide, i, zoom_lvl))
    return slides

class Slide:
    'combines all functions concerning EM and FM image fetching'
    def __init__(self, training_dir, slide_subfolders, count, zoom_lvl):
        'initialize'        
        self.count = count 
        self.zoom_lvl = zoom_lvl
        self.EM_dir = training_dir + slide_subfolders["EM"]      #directory of EM slide
        self.FM_dir = training_dir + slide_subfolders["FM"]      #directory of FM slide
        
        self.slideXYList = self.getXYList()      #list of xy coordinates of EM slide
        self.slideXYList_FM = self.getXYList_FM()      #list of xy coordinates of FM slide
        self.n_tiles = self.slideXYList[-1]       #number of tiles in columns and rows (not perfect)
        
        self.max_tiles = self.maxTiles()      #max number of tiles in columns and rows (perfect)
        
    def xyFromFilename(self, f):
        'returns x and y coordinates of slides in f'
        slide_parts = f.split("_")
        return (int(slide_parts[0]), int(slide_parts[1]), self.count)
   
    def getXYList(self):
        'returns x and y coordinates of EM slides' 
        return [Slide.xyFromFilename(self, f) for f in os.listdir(self.EM_dir) if f.endswith(str(self.zoom_lvl)+".png")]
      
    def getXYList_FM(self):
        'returns x and y coordinates of EM slides' 
        return [Slide.xyFromFilename(self, f) for f in os.listdir(self.FM_dir) if f.endswith("5.png")]    
    
    def maxTiles(self):
        'returns max # of tiles in width and height'
        max_i = max(self.slideXYList,key=lambda item:item[0])[0]
        max_j = max(self.slideXYList,key=lambda item:item[1])[1]
        return np.array([max_i,max_j])    
    
    def getTile(self, tile_xy, EM_case):
        'read files'
        if EM_case == True:
            tile_name = "{}_{}_{}.png".format(tile_xy[0], tile_xy[1], self.zoom_lvl)
            filename = self.EM_dir + tile_name
            im_tile = cv2.imread(filename)
            if im_tile is None:
                im_tile = np.ones((1024,1024))*255
            else:
                im_tile = im_tile[:, :, 1]  # use the first channel (second and third are the same)
        else:
            tile_name = "{}_{}_{}.png".format(tile_xy[0], tile_xy[1], self.zoom_lvl) 
            filename = self.FM_dir + tile_name
            im_tile = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            im_tile = im_tile[:, :, 3]  # use the alpha channel
        return im_tile
    
    def createPatch(self, tile_size, tile_xy, patch_size, cut_on, coords, EM_case):
        #print('JOEEEE')
        'Returns patch'
        if ((coords[0]%tile_size) + patch_size[0]) / (2*tile_size) >= 1 and ((coords[1]%tile_size) + patch_size[1]) / (2*tile_size) >= 1:
            im_tile_0 = self.getTile((tile_xy[0]+0, tile_xy[1]+0), EM_case)
            im_tile_1 = self.getTile((tile_xy[0]+1, tile_xy[1]+0), EM_case)
            im_tile_2 = self.getTile((tile_xy[0]+2, tile_xy[1]+0), EM_case)
            im_tile_3 = self.getTile((tile_xy[0]+0, tile_xy[1]+1), EM_case)
            im_tile_4 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
            im_tile_5 = self.getTile((tile_xy[0]+2, tile_xy[1]+1), EM_case)
            im_tile_6 = self.getTile((tile_xy[0]+0, tile_xy[1]+2), EM_case)
            im_tile_7 = self.getTile((tile_xy[0]+1, tile_xy[1]+2), EM_case)
            im_tile_8 = self.getTile((tile_xy[0]+2, tile_xy[1]+2), EM_case)
        
        
            coords2 = (coords[0] + patch_size[0]) % tile_size, (coords[1] + patch_size[1]) % tile_size
#             print(patch_size)
#             print(coords[0]+patch_size[1], coords[1]+patch_size[1])
#             print(coords2[0], coords2[1])
            
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]:  ], 
                                         im_tile_1[:          , coords[1]:  ],
                                         im_tile_2[:coords2[0], coords[1]:  ]), axis=0)
            im_patch_1 = np.concatenate((im_tile_3[coords[0]: , :           ], 
                                         im_tile_4[:          , :           ],
                                         im_tile_5[:coords2[0], :           ]), axis=0)
            im_patch_2 = np.concatenate((im_tile_6[coords[0]: , :coords2[1] ], 
                                         im_tile_7[:          , :coords2[1] ],
                                         im_tile_8[:coords2[0], :coords2[1] ]), axis=0)
            im_patch = np.concatenate((im_patch_0,
                                       im_patch_1,
                                       im_patch_2), axis=1)
            im_patch = im_patch[cut_on:,cut_on:]
            #print('case 1', im_patch.shape)
        
        elif ((coords[0]%tile_size) + patch_size[1]) / (2*tile_size) >= 1:
            im_tile_0 = self.getTile((tile_xy[0]+0, tile_xy[1]+0), EM_case)
            im_tile_1 = self.getTile((tile_xy[0]+1, tile_xy[1]+0), EM_case)
            im_tile_2 = self.getTile((tile_xy[0]+2, tile_xy[1]+0), EM_case)
            im_tile_3 = self.getTile((tile_xy[0]+0, tile_xy[1]+1), EM_case)
            im_tile_4 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
            im_tile_5 = self.getTile((tile_xy[0]+2, tile_xy[1]+1), EM_case)
            
            coords2 = (coords[0] + patch_size[0]) % tile_size, (coords[1] + patch_size[1]) % tile_size
            
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]: ], 
                                         im_tile_1[:          , coords[1]: ],
                                         im_tile_2[:coords2[0], coords[1]: ]), axis=0)
            im_patch_1 = np.concatenate((im_tile_3[coords[0]: , :coords2[1]], 
                                         im_tile_4[:          , :coords2[1]],
                                         im_tile_5[:coords2[0], :coords2[1]]), axis=0)
            im_patch = np.concatenate((im_patch_0,
                                       im_patch_1), axis=1)
            im_patch = im_patch[cut_on:,cut_on:]
            #print('case x', im_patch.shape)
        
        elif ((coords[1]%tile_size) + patch_size[1]) / (2*tile_size) >= 1:
            im_tile_0 = self.getTile((tile_xy[0]+0, tile_xy[1]+0), EM_case)
            im_tile_1 = self.getTile((tile_xy[0]+1, tile_xy[1]+0), EM_case)
            im_tile_3 = self.getTile((tile_xy[0]+0, tile_xy[1]+1), EM_case)
            im_tile_4 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
            im_tile_6 = self.getTile((tile_xy[0]+0, tile_xy[1]+2), EM_case)
            im_tile_7 = self.getTile((tile_xy[0]+1, tile_xy[1]+2), EM_case)
            
            coords2 = (coords[0] + patch_size[0]) % tile_size, (coords[1] + patch_size[1]) % tile_size
            
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]:  ], 
                                         im_tile_1[:coords2[0], coords[1]:  ]), axis=0)
            im_patch_1 = np.concatenate((im_tile_3[coords[0]: , :           ], 
                                         im_tile_4[:coords2[0], :           ]), axis=0)
            im_patch_2 = np.concatenate((im_tile_6[coords[0]: , :coords2[1]], 
                                         im_tile_7[:coords2[0], :coords2[1]]), axis=0)
            im_patch = np.concatenate((im_patch_0,
                                       im_patch_1,
                                       im_patch_2), axis=1)
            im_patch = im_patch[cut_on:,cut_on:]
            #print('case y', im_patch.shape)
        
        else: #(coords[0] + patch_size) > tile_size and (coords[1] + patch_size) > tile_size:
            #get all adjacent tiles
            im_tile_0 = self.getTile((tile_xy[0], tile_xy[1]), EM_case)
            im_tile_1 = self.getTile((tile_xy[0]+1, tile_xy[1]), EM_case)
            im_tile_2 = self.getTile((tile_xy[0], tile_xy[1]+1), EM_case)
            im_tile_3 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
            #new coordinates in adjacent tiles
            coords2 = (coords[0] + patch_size[0]) % tile_size, (coords[1] + patch_size[1]) % tile_size
            #concatenate patches
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]: ],
                                         im_tile_1[:coords2[0], coords[1]: ]), axis=0)
            im_patch_1 = np.concatenate((im_tile_2[coords[0]: , :coords2[1]],
                                         im_tile_3[:coords2[0], :coords2[1]]), axis=0)
            im_patch = np.concatenate((im_patch_0, im_patch_1), axis=1)
            im_patch = im_patch[cut_on:,cut_on:]
            #print('case 4', im_patch.shape)
        return im_patch
    
##### Catmaid #####

def catmaid(predictions, testslide_subfolders, fig_size, fig_name):
    
    #initialization
#     slides = getSlideList(testslide_subfolders, training_dir, 0)[0]
#     x_set = slides.slideXYList
#     x_set_max = slides.n_tiles
    slide_fm = np.zeros(((EM_tile_size*6),(EM_tile_size*6)))
    
    #figure (FM prediction)
    slide_test = predictions    
    
    #figure (FM)
    for i in range(2):
        for j in range(2):
            tile_name = "{}_{}_{}.png".format(i, j, 5) 
            filename = training_dir + testslide_subfolders[0]["FM"] + tile_name
            im_tile = cv2.imread(filename)
            if im_tile is None:
                im_tile = np.ones((1024,1024))*255
            else:
                im_tile = im_tile[:, :, 1]  # use the first channel (second and third are the same)
            slide_fm[1024*i:1024*(i+1), 1024*j:(j+1)*1024] = im_tile/255
    
    #get EM slides for different zoom levels
    em_slides = []
    zoom_lvl = [5,4,3,2,1,0]
    for k in range(len(zoom_lvl)):
        em_slides.extend(getSlideList(testslide_subfolders, training_dir, zoom_lvl[k]))
    
    #adjust fm images to transparent images
    slide_test_grey = rescale_intensity(slide_test, out_range=np.uint8)
    slide_test_rgb = grey2rgb(np.array(slide_test_grey, dtype=np.uint8), alpha=True)
    slide_fm_grey = rescale_intensity(slide_fm, out_range=np.uint8)
    slide_fm_rgb = grey2rgb(np.array(slide_fm_grey, dtype=np.uint8), alpha=True)

    #set color
    slide_test_rgb[:,:,0] = 0
    slide_test_rgb[:,:,2] = 0
    slide_fm_rgb[:,:,1] = 0
    
    #set image pixel size
    cut_on=0
    cut_off = 1024,1024
    
    def transparent(trans_pred, trans_real, trans_par1, trans_par2, zoom, x_slid, y_slid, save_img): 
        #invert transparency
        tran_1 = 255-trans_pred/100*255
        trans_par1[:,:,3] = tran_1
        tran_2 = 255-trans_real/100*255
        trans_par2[:,:,3] = tran_2
        
        #create em image
        coords = cut_off[0]*x_slid/100*2**zoom, cut_off[1]*y_slid/100*2**zoom
        tile_xy = int(coords[0] / EM_tile_size), int(coords[1] / EM_tile_size)
        coords_em = int(coords[0] - (EM_tile_size*tile_xy[0])), int(coords[1] - (EM_tile_size*tile_xy[1]))
        coords_fm = coords[0]/(2**zoom), coords[1]/(2**zoom)
        EM_case = True
        slide_em_grey = em_slides[zoom].createPatch(EM_tile_size, tile_xy, cut_off, cut_on, coords_em, EM_case)
        
        #create fm images
        trans_par2 = trans_par2[int(coords_fm[0]):int(coords_fm[0]+cut_off[0]/(2**zoom)),
                                int(coords_fm[1]):int(coords_fm[1]+cut_off[1]/(2**zoom))]
        trans_par1 = trans_par1[int(coords_fm[0]):int(coords_fm[0]+cut_off[0]/(2**zoom)),
                                int(coords_fm[1]):int(coords_fm[1]+cut_off[1]/(2**zoom))]
        real_fm = np.repeat(np.repeat(trans_par2, 2**zoom, axis=1), 2**zoom, axis=0)
        pred_fm = np.repeat(np.repeat(trans_par1, 2**zoom, axis=1), 2**zoom, axis=0)
        
        #calculate things
        #accuracy:
        
        #pearson:
        def calc_pearson(y_true2, y_pred2):
            #doesnt work yet, and will prob not due to differences in length/height
            #y_true = rescale_intensity(y_true, out_range=np.uint8)
            #y_pred = rescale_intensity(y_pred, out_range=np.uint8)
            y_true = y_true2[:,:,:3]
            y_pred = y_pred2[:,:,:3]
            y_true_mean = y_true-np.mean(y_true)
            y_pred_mean = y_pred-np.mean(y_pred)
            return np.sum(y_true_mean*y_pred_mean) / np.sqrt(np.sum((y_true_mean)**2)*np.sum((y_pred_mean)**2))
        #pearson_coeff = calc_pearson(real_fm, pred_fm)
        #print(pearson_coeff)
        
        #plot
        fig = plt.figure(figsize=(fig_size,fig_size))
        plt.imshow(slide_em_grey, cmap='Greys')
        plt.imshow(pred_fm)
        plt.imshow(real_fm)
        plt.axis('off')
        if save_img == True:
            filename = os.path.join("figures", fig_name+'_'+str(int(trans_pred))+
                                    'Tpred_'+str(int(trans_real))+'Treal_'+
                                    str(int(zoom))+'Zoom_'+str(int(y_slid))+'-'+str(int(x_slid))+'Pos.png')
            fig.savefig(filename)
            #imsave(filename, tst)
     
    #widget info
    button = wg.Button(description="Save")
    button_output = wg.Output()
    
    slider1 = wg.IntSlider(min=0, max=100, step=1, value=100)
    slider2 = wg.IntSlider(min=0, max=100, step=1, value=100) 
    slider3 = wg.IntSlider(min=0, max=5, step=1, value=0)
    
    slider4 = wg.IntSlider(min=0, max=120, step=1, value=0, orientation='horizontal',readout=True)
    slider5 = wg.IntSlider(min=0, max=120, step=1, value=0, readout=True)
    
#     slider6 = wg.IntSlider(min=0, max=100, step=1, value=100) 
#     slider7 = wg.IntSlider(min=0, max=100, step=1, value=100) 
#     slider8 = wg.IntSlider(min=0, max=100, step=1, value=100) 
    
    slider1.style.handle_color = 'lightblue'
    slider2.style.handle_color = 'purple'
    
    slid_1_text = wg.HBox([slider1, wg.Label('Transparency of FM (predicted)')])
    slid_2_text = wg.HBox([slider2, wg.Label('Transparency of FM (real)')])
    slid_3_text = wg.HBox([slider3, wg.Label('Zoom')])
    slid_4_text = wg.HBox([slider4, wg.Label('Position (y)')])
    slid_5_text = wg.HBox([slider5, wg.Label('Position (x)')])
    
    slids_out = wg.VBox([slid_1_text, slid_2_text, slid_3_text, slid_5_text, slid_4_text])
   
    func_out = wg.interactive_output(transparent, 
                                     {'trans_pred': slider1,
                                     'trans_real': slider2,
                                     'trans_par1': wg.fixed(slide_test_rgb),
                                     'trans_par2': wg.fixed(slide_fm_rgb),
                                     'zoom': slider3,
                                     'x_slid': slider4,
                                     'y_slid': slider5,
                                     'save_img': wg.fixed(False)})
    def test(b):
        transparent(slider1.value, 
                    slider2.value, 
                    slide_test_rgb, 
                    slide_fm_rgb, 
                    slider3.value, 
                    slider4.value, 
                    slider5.value,
                    True)
        filename = os.path.join("figures", fig_name+'_'+str(int(slider1.value))+'Tpred_'+str(int(slider2.value))+'Treal_'+
                                    str(int(slider3.value))+'Zoom_'+str(int(slider5.value))+'-'+str(int(slider4.value))+'Pos.png')
        with button_output:
            print("Image saved as: ", filename)
    
    button.on_click(test)

    display(button, button_output, slids_out, func_out)