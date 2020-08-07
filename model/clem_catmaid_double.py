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
from skimage import (io, color, img_as_float, util, exposure, img_as_ubyte)
import ipywidgets as wg

from xlutils.copy import copy    
from xlrd import open_workbook

##### General variables #####

def variables(training_dir_0, EM_tile_size_0):
    
    global training_dir
    global EM_tile_size
    
    training_dir = training_dir_0
    EM_tile_size = EM_tile_size_0
        
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
        #self.FM_dir = training_dir + slide_subfolders["FM"]      #directory of FM slide
        self.FM_dir_hoechst = training_dir + slide_subfolders["FM_1"]
        self.FM_dir_insulin = training_dir + slide_subfolders["FM_2"]
        
        self.slideXYList_tot = self.getXYList()      #list of xy coordinates of every EM_0 slide
        self.slideXYList = self.slideXYList_tot      #just same as slideXYList_tot
        
    def xyFromFilename(self, f):
        'returns x and y coordinates of slides in f'
        slide_parts = f.split("_")
        return (int(slide_parts[0]), int(slide_parts[1]), self.count)
   
    def getXYList(self):
        'returns x and y coordinates of EM slides' 
        return [Slide.xyFromFilename(self, f) for f in os.listdir(self.EM_dir) if f.endswith(str(self.zoom_lvl)+".png")]
      
    def getTile(self, tile_xy, EM_case):
        'read files'
        if EM_case == True:
            tile_name = "{}_{}_{}.png".format(tile_xy[0], tile_xy[1], self.zoom_lvl)
            filename = self.EM_dir + tile_name
            im_tile = img_as_float(io.imread(filename))
            if im_tile is None:
                rgba = np.zeros((1024,1024,4))*255
            else:
                inverted = util.invert(im_tile)
                rgba = color.grey2rgb(inverted, alpha = True)  
        if EM_case == 'FM_hoechst':
            tile_name = "{}_{}_{}.png".format(tile_xy[0], tile_xy[1], self.zoom_lvl) 
            filename = self.FM_dir_hoechst + tile_name
            im_tile = img_as_float(io.imread(filename))
            if im_tile is None:
                rgba = np.zeros((1024,1024,4))*255
            else:
                rgba = color.grey2rgb(im_tile, alpha=True)
        if EM_case == 'FM_insulin':
            tile_name = "{}_{}_{}.png".format(tile_xy[0], tile_xy[1], self.zoom_lvl) 
            filename = self.FM_dir_insulin + tile_name
            im_tile = img_as_float(io.imread(filename))
            if im_tile is None:
                rgba = np.zeros((1024,1024,4))*255
            else:
                rgba = color.grey2rgb(im_tile, alpha=True)
        return rgba
    
    def createPatch(self, tile_size, tile_xy, patch_size, coords, EM_case):
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
            
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]:  ,:], 
                                         im_tile_1[:          , coords[1]:  ,:],
                                         im_tile_2[:coords2[0], coords[1]:  ,:]), axis=0)
            im_patch_1 = np.concatenate((im_tile_3[coords[0]: , :           ,:], 
                                         im_tile_4[:          , :           ,:],
                                         im_tile_5[:coords2[0], :           ,:]), axis=0)
            im_patch_2 = np.concatenate((im_tile_6[coords[0]: , :coords2[1] ,:], 
                                         im_tile_7[:          , :coords2[1] ,:],
                                         im_tile_8[:coords2[0], :coords2[1] ,:]), axis=0)
            im_patch = np.concatenate((im_patch_0,
                                       im_patch_1,
                                       im_patch_2), axis=1)
            #im_patch = im_patch[cut_on:,cut_on:]
            #print('case 1', im_patch.shape)
        
        elif ((coords[0]%tile_size) + patch_size[1]) / (2*tile_size) >= 1:
            im_tile_0 = self.getTile((tile_xy[0]+0, tile_xy[1]+0), EM_case)
            im_tile_1 = self.getTile((tile_xy[0]+1, tile_xy[1]+0), EM_case)
            im_tile_2 = self.getTile((tile_xy[0]+2, tile_xy[1]+0), EM_case)
            im_tile_3 = self.getTile((tile_xy[0]+0, tile_xy[1]+1), EM_case)
            im_tile_4 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
            im_tile_5 = self.getTile((tile_xy[0]+2, tile_xy[1]+1), EM_case)
            
            coords2 = (coords[0] + patch_size[0]) % tile_size, (coords[1] + patch_size[1]) % tile_size
            
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]: ,:], 
                                         im_tile_1[:          , coords[1]: ,:],
                                         im_tile_2[:coords2[0], coords[1]: ,:]), axis=0)
            im_patch_1 = np.concatenate((im_tile_3[coords[0]: , :coords2[1],:], 
                                         im_tile_4[:          , :coords2[1],:],
                                         im_tile_5[:coords2[0], :coords2[1],:]), axis=0)
            im_patch = np.concatenate((im_patch_0,
                                       im_patch_1), axis=1)
            #im_patch = im_patch[cut_on:,cut_on:]
            #print('case x', im_patch.shape)
        
        elif ((coords[1]%tile_size) + patch_size[1]) / (2*tile_size) >= 1:
            im_tile_0 = self.getTile((tile_xy[0]+0, tile_xy[1]+0), EM_case)
            im_tile_1 = self.getTile((tile_xy[0]+1, tile_xy[1]+0), EM_case)
            im_tile_3 = self.getTile((tile_xy[0]+0, tile_xy[1]+1), EM_case)
            im_tile_4 = self.getTile((tile_xy[0]+1, tile_xy[1]+1), EM_case)
            im_tile_6 = self.getTile((tile_xy[0]+0, tile_xy[1]+2), EM_case)
            im_tile_7 = self.getTile((tile_xy[0]+1, tile_xy[1]+2), EM_case)
            
            coords2 = (coords[0] + patch_size[0]) % tile_size, (coords[1] + patch_size[1]) % tile_size
            
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]:  ,:], 
                                         im_tile_1[:coords2[0], coords[1]:  ,:]), axis=0)
            im_patch_1 = np.concatenate((im_tile_3[coords[0]: , :           ,:], 
                                         im_tile_4[:coords2[0], :           ,:]), axis=0)
            im_patch_2 = np.concatenate((im_tile_6[coords[0]: , :coords2[1],:], 
                                         im_tile_7[:coords2[0], :coords2[1],:]), axis=0)
            im_patch = np.concatenate((im_patch_0,
                                       im_patch_1,
                                       im_patch_2), axis=1)
            #im_patch = im_patch[cut_on:,cut_on:]
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
            im_patch_0 = np.concatenate((im_tile_0[coords[0]: , coords[1]: ,:],
                                         im_tile_1[:coords2[0], coords[1]: ,:]), axis=0)
            im_patch_1 = np.concatenate((im_tile_2[coords[0]: , :coords2[1],:],
                                         im_tile_3[:coords2[0], :coords2[1],:]), axis=0)
            im_patch = np.concatenate((im_patch_0, im_patch_1), axis=1)
            #im_patch = im_patch[cut_on:,cut_on:]
            #print('case 4', im_patch.shape)
        return im_patch

def pearson(y_true, y_pred):
    'pearson correlation coefficient'
    x0 = y_true-np.mean(y_true)
    y0 = y_pred-np.mean(y_pred) 
    return np.sum(x0*y0) / np.sqrt(np.sum((x0)**2)*np.sum((y0)**2))
  
## Transparancy and color matrices
t_hoechst = [[0.1, 0.0, 0.0, 0.1],
            [0.0, 0.1, 0.0, 0.1],
            [0.0, 0.0, 0.9, 0.9],
            [0.0, 0.0, 0.0, 0.0]]
t_insulin = [[0.9, 0.0, 0.0, 0.9],
            [0.0, 0.6, 0.0, 0.6],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]
t_h_pred = [[0.9, 0.0, 0.0, 0.9],
            [0.0, 0.1, 0.0, 0.1],
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0]]
t_i_pred = [[0.1, 0.0, 0.0, 0.1],
            [0.0, 0.9, 0.0, 0.9],
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0]]
    
def catmaid(testslide_subfolders, pred_hoechst, pred_insulin, cut_off, fig_size, fig_name, hoechst_pred, insulin_pred):
    
    ##initialization
    #set height, width of image
    cut_off = cut_off, cut_off
    
    #get EM & FM slides for different zoom levels
    slides = []
    zoom_lvl = [5,4,3,2,1,0]
    for k in range(len(zoom_lvl)):
        slides.extend(getSlideList(testslide_subfolders, training_dir, zoom_lvl[k]))
    
    #get FM hoechst NOT upsampled
    slide_hoechst_rgb = np.zeros(((EM_tile_size*3),(EM_tile_size*3),4))  
    for i in range(2):
        for j in range(2):
            tile_name = "{}_{}_{}.png".format(i, j, 5) 
            filename = training_dir + testslide_subfolders[0]["FM_1"] + tile_name
            im_tile = img_as_float(io.imread(filename))
            if im_tile is None:
                im_tile = np.zeros((1024,1024,4))*255
            else:
                im_tile = color.grey2rgb(im_tile, alpha=True)
            slide_hoechst_rgb[1024*i:1024*(i+1), 1024*j:(j+1)*1024,:] = im_tile/255
    slide_hoechst_rgb = np.dot(slide_hoechst_rgb, t_hoechst)
    slide_hoechst_rgb = exposure.rescale_intensity(slide_hoechst_rgb)
    
    #get FM insulin NOT upsampled
    slide_insulin_rgb = np.zeros(((EM_tile_size*3),(EM_tile_size*3),4))  
    for i in range(2):
        for j in range(2):
            tile_name = "{}_{}_{}.png".format(i, j, 5) 
            filename = training_dir + testslide_subfolders[0]["FM_2"] + tile_name
            im_tile = img_as_float(io.imread(filename))
            if im_tile is None:
                im_tile = np.zeros((1024,1024,4))*255
            else:
                im_tile = color.grey2rgb(im_tile, alpha=True)
            slide_insulin_rgb[1024*i:1024*(i+1), 1024*j:(j+1)*1024,:] = im_tile/255
    slide_insulin_rgb = np.dot(slide_insulin_rgb, t_insulin)
    slide_insulin_rgb = exposure.rescale_intensity(slide_insulin_rgb)
    
    #Hoechst prediction
    if hoechst_pred == True:
        predictions = np.load('predictions/' + pred_hoechst + '.npy')
        # set color of prediction
        predictions_grey = exposure.rescale_intensity(predictions, out_range=np.uint8)
        pred_rgb_hoechst = color.grey2rgb(np.array(predictions_grey, dtype=np.uint8), alpha=True)  
        pred_rgb_hoechst = np.dot(pred_rgb_hoechst, t_h_pred)
        pred_rgb_hoechst = exposure.rescale_intensity(pred_rgb_hoechst)
    else:
        pred_rgb_hoechst = 0
        
    #Insulin prediction   
    if insulin_pred == True:
        # load prediction
        predictions = np.load('predictions/' + pred_insulin + '.npy')
        # set color of prediction
        predictions_grey = exposure.rescale_intensity(predictions, out_range=np.uint8)
        pred_rgb_insulin = color.grey2rgb(np.array(predictions_grey, dtype=np.uint8), alpha=True)  
        pred_rgb_insulin = np.dot(pred_rgb_insulin, t_i_pred)
        pred_rgb_insulin = exposure.rescale_intensity(pred_rgb_insulin)
    else:
        pred_rgb_insulin = 0   
    
    def transparent(trans_hoechst, trans_insulin, trans_hoechst_real, trans_insulin_real, pred_rgb_hoechst, pred_rgb_insulin, zoom, x_slid, y_slid, save_img, hoechst_pearson, insulin_pearson):
 
        #create em image
        coords_img = cut_off[0]*x_slid/100*2**zoom, cut_off[1]*y_slid/100*2**zoom
        tile_xy = int(coords_img[0] / EM_tile_size), int(coords_img[1] / EM_tile_size)
        coords = int(coords_img[0] - (EM_tile_size*tile_xy[0])), int(coords_img[1] - (EM_tile_size*tile_xy[1]))
        slide_em_grey = slides[zoom].createPatch(EM_tile_size, tile_xy, cut_off, coords, EM_case=True)

        #creat NON upsampled
        coords_fm = coords_img[0]/(2**zoom), coords_img[1]/(2**zoom)
        slide_hoechst_rgb_2 = slide_hoechst_rgb[int(coords_fm[0]):int(coords_fm[0]+cut_off[0]/(2**zoom)),
                                int(coords_fm[1]):int(coords_fm[1]+cut_off[1]/(2**zoom))]
        slide_fm_hoechst= np.repeat(np.repeat(slide_hoechst_rgb_2, 2**zoom, axis=1), 2**zoom, axis=0)

        #creat NON upsampled
        slide_insulin_rgb_2 = slide_insulin_rgb[int(coords_fm[0]):int(coords_fm[0]+cut_off[0]/(2**zoom)),
                                int(coords_fm[1]):int(coords_fm[1]+cut_off[1]/(2**zoom))]
        slide_fm_insulin= np.repeat(np.repeat(slide_insulin_rgb_2, 2**zoom, axis=1), 2**zoom, axis=0)
        
        #select area prediction 
        coords_fm = coords_img[0]/(2**zoom), coords_img[1]/(2**zoom)
        if hoechst_pred == True:
            pred_rgb_hoechst = pred_rgb_hoechst[int(coords_fm[0]):int(coords_fm[0]+cut_off[0]/(2**zoom)),
                                int(coords_fm[1]):int(coords_fm[1]+cut_off[1]/(2**zoom))]
            pred_rgb_hoechst = np.repeat(np.repeat(pred_rgb_hoechst, 2**zoom, axis=1), 2**zoom, axis=0)
        if insulin_pred == True:
            pred_rgb_insulin = pred_rgb_insulin[int(coords_fm[0]):int(coords_fm[0]+cut_off[0]/(2**zoom)),
                                int(coords_fm[1]):int(coords_fm[1]+cut_off[1]/(2**zoom))]
            pred_rgb_insulin = np.repeat(np.repeat(pred_rgb_insulin, 2**zoom, axis=1), 2**zoom, axis=0)
        
        if hoechst_pred == True and insulin_pred == True:
            images = []
            images.append(slide_em_grey)
            images.append(slide_fm_hoechst*trans_hoechst_real/100)
            images.append(slide_fm_insulin*trans_insulin_real/100)
            images.append(pred_rgb_hoechst*trans_hoechst/100)
            images.append(pred_rgb_insulin*trans_insulin/100)
            tst = exposure.rescale_intensity(np.sum(images, axis=0), in_range=(0,1))
        elif hoechst_pred == True and insulin_pred == False:            
            images = []
            images.append(slide_em_grey)
            images.append(slide_fm_hoechst*trans_hoechst_real/100)
            images.append(slide_fm_insulin*trans_insulin_real/100)
            images.append(pred_rgb_hoechst*trans_hoechst/100)
            tst = exposure.rescale_intensity(np.sum(images, axis=0), in_range=(0,1))
        elif hoechst_pred == False and insulin_pred == True:          
            images = []
            images.append(slide_em_grey)
            images.append(slide_fm_hoechst*trans_hoechst_real/100)
            images.append(slide_fm_insulin*trans_insulin_real/100)
            images.append(pred_rgb_insulin*trans_insulin/100)
            tst = exposure.rescale_intensity(np.sum(images, axis=0), in_range=(0,1))
        else:
            images = []
            images.append(slide_em_grey)
            images.append(slide_fm_hoechst*trans_hoechst_real/100)
            images.append(slide_fm_insulin*trans_insulin_real/100)
#            print(np.shape(images[0]))
#             print(np.shape(images[1]))
#             print(np.shape(images[2]))
            tst = exposure.rescale_intensity(np.sum(images, axis=0), in_range=(0,1))
        
        #plot
        if save_img:
            filename = os.path.join("figures", fig_name+'_'+str(int(trans_hoechst))+
                                    'Tpred_'+str(int(trans_insulin))+'Treal_'+
                                    str(int(zoom))+'Zoom_'+str(int(y_slid))+'-'+str(int(x_slid))+'Pos.png')
            #print('jpoe')
            #fig.savefig(filename)
#             imsave(filename, rescale(tst, 4))
            io.imsave(filename, img_as_ubyte(tst))
        
        fig = plt.figure(figsize=(fig_size,fig_size))
        plt.imshow(tst)
        plt.axis('off')       
        
        if hoechst_pearson == True:
            book_ro = open_workbook("PearsonCorrelationHoechst.xls")
            book = copy(book_ro)  # creates a writeable copy
            sheet1 = book.get_sheet(0)  # get a first sheet
            
            count = int(book_ro.sheet_by_index(0).cell(1,7).value)
            sheet1.write(count,0, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            sheet1.write(count,1, str(training_dir + testslide_subfolders[0]["FM_1"]))
            sheet1.write(count,2, pred_hoechst)
            sheet1.write(count,3, y_slid)
            sheet1.write(count,4, x_slid)
            sheet1.write(count,5, zoom)
            sheet1.write(count,6, pearson(slide_fm_hoechst[:,:,2], pred_rgb_hoechst[:,:,0]))
            sheet1.write(1,7, count+1)
            book.save("PearsonCorrelationHoechst.xls")
            
            with button_output:
                print('Pearson correlation (hoechst):', pearson(slide_fm_hoechst[:,:,2], pred_rgb_hoechst[:,:,0]))
        if insulin_pearson == True:
            book_ro = open_workbook("PearsonCorrelationInsulin.xls")
            book = copy(book_ro)  # creates a writeable copy
            sheet1 = book.get_sheet(0)  # get a first sheet
            
            count = int(book_ro.sheet_by_index(0).cell(1,7).value)
            sheet1.write(count,0, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            sheet1.write(count,1, str(training_dir + testslide_subfolders[0]["FM_2"]))
            sheet1.write(count,2, pred_insulin)
            sheet1.write(count,3, y_slid)
            sheet1.write(count,4, x_slid)
            sheet1.write(count,5, zoom)
            sheet1.write(count,6, pearson(slide_fm_insulin[:,:,0], pred_rgb_insulin[:,:,1]))
            sheet1.write(1,7, count+1)
            book.save("PearsonCorrelationInsulin.xls")
            with button_output:
                print('Pearson correlation (insulin):', pearson(slide_fm_insulin[:,:,0], pred_rgb_insulin[:,:,1]))
    
    #widget info
    button_save = wg.Button(description="Save")
    button_hoechst_pearson = wg.Button(description="Pearson (hoechst)")
    button_insulin_pearson = wg.Button(description="Pearson (insulin)")
    button_output = wg.Output()
    
    slider1 = wg.IntSlider(min=1, max=100, step=1, value=1)
    slider2 = wg.IntSlider(min=1, max=100, step=1, value=1) 
    slider3 = wg.IntSlider(min=1, max=100, step=1, value=61)
    slider4 = wg.IntSlider(min=1, max=100, step=1, value=61)
    
    slider1.style.handle_color = 'red'
    slider2.style.handle_color = 'green'
    slider3.style.handle_color = 'blue'
    slider4.style.handle_color = 'yellow'
    
    slider5 = wg.IntSlider(min=0, max=5, step=1, value=0)
    slider6 = wg.IntSlider(min=0, max=120, step=1, value=12, orientation='horizontal',readout=True)
    slider7 = wg.IntSlider(min=0, max=120, step=1, value=12, readout=True)    
    
    slid_1_text = wg.HBox([slider1, wg.Label('Transparency of hoechst (predicted)')])
    slid_2_text = wg.HBox([slider2, wg.Label('Transparency of insulin (predicted)')])
    slid_3_text = wg.HBox([slider3, wg.Label('Transparency of hoechst (real)')])
    slid_4_text = wg.HBox([slider4, wg.Label('Transparency of insulin (real)')])

    slid_5_text = wg.HBox([slider5, wg.Label('Zoom')])
    slid_6_text = wg.HBox([slider6, wg.Label('Position (y)')])
    slid_7_text = wg.HBox([slider7, wg.Label('Position (x)')])
    
    if hoechst_pred == True and insulin_pred == True:
        slids_out = wg.VBox([slid_1_text, slid_2_text, slid_3_text, slid_4_text, slid_5_text, slid_7_text, slid_6_text])
        buttons = wg.HBox([button_save, button_hoechst_pearson, button_insulin_pearson])
    if hoechst_pred == False and insulin_pred == True:
        slids_out = wg.VBox([slid_2_text, slid_3_text, slid_4_text, slid_5_text, slid_7_text, slid_6_text])
        buttons = wg.HBox([button_save, button_insulin_pearson])
    if hoechst_pred == True and insulin_pred == False:
        slids_out = wg.VBox([slid_1_text, slid_3_text, slid_4_text, slid_5_text, slid_7_text, slid_6_text])
        buttons = wg.HBox([button_save, button_hoechst_pearson])
    if hoechst_pred == False and insulin_pred == False:
        slids_out = wg.VBox([slid_3_text, slid_4_text, slid_5_text, slid_7_text, slid_6_text])
        buttons = wg.HBox([button_save])
      
    func_out = wg.interactive_output(transparent, 
                                         {'trans_hoechst': slider1,
                                         'trans_insulin': slider2,
                                         'trans_hoechst_real': slider3,
                                         'trans_insulin_real': slider4,
                                         'pred_rgb_hoechst': wg.fixed(pred_rgb_hoechst),
                                         'pred_rgb_insulin': wg.fixed(pred_rgb_insulin),
                                         'zoom': slider5,
                                         'x_slid': slider6,
                                         'y_slid': slider7,
                                         'save_img': wg.fixed(False),
                                         'hoechst_pearson': wg.fixed(False),
                                         'insulin_pearson': wg.fixed(False)})
    def test(b):
        transparent(slider1.value, slider2.value, slider3.value, slider4.value,
                        pred_rgb_hoechst, pred_rgb_insulin, 
                        slider5.value, slider6.value, slider7.value,
                        True, False, False)
        filename = os.path.join("figures", fig_name+'_'+str(int(slider1.value))+'Thoechst_'+str(int(slider2.value))+'Tinsulin_'+
                                        str(int(slider5.value))+'Zoom_'+str(int(slider7.value))+'-'+str(int(slider6.value))+'Pos.png')
        with button_output:
            print("Image saved as: ", filename)
    def pearson_act_hoechst(b):
        transparent(slider1.value, slider2.value, slider3.value, slider4.value,
                        pred_rgb_hoechst, pred_rgb_insulin, 
                        slider5.value, slider6.value, slider7.value,
                        False, True, False)
    def pearson_act_insulin(b):
        transparent(slider1.value, slider2.value, slider3.value, slider4.value,
                        pred_rgb_hoechst, pred_rgb_insulin, 
                        slider5.value, slider6.value, slider7.value,
                        False, False, True)
    
    button_save.on_click(test)
    button_hoechst_pearson.on_click(pearson_act_hoechst)
    button_insulin_pearson.on_click(pearson_act_insulin)
    
    display(buttons, button_output, slids_out, func_out)
    
    

    