import cv2
import glob
import os
import numpy as np
#import argparse
#import re

# Generating data for pix2pix model for Arnoldifier
# By Twenkid/Todor Arnaudov, MIT license
# http://github.com/twenkid
# http://twenkid.com
# http://artificial-mind.blogspot.com
# It's a part from a bigger file face_montage for various other utility functions
# Takes a directory of aligned color faces, extracted for the training of the Deepfacelab model
# and combines them into images with two faces, one grayscale, one the original color, next to another,
# in order to be used as a dataset by the pix2pix training routing.
# The pix2pix training script in this repo is tailored for 128x128 (Geforce 750 Ti training on the GPU)
# https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/DeepFaceLab_DirectX12/_internal/DeepFaceLab/colorize/pix_128.py
# If you want to change it, adjust w, h ... or send it as parameter from the call resize=(256,256) etc. instead of (128,128) 

#to_grayscale_global = True #for turnoff - when resizing/gaussian starts to colorize a little! --> future; for now -- with Saturation effect in the editor/Openshot (I shouldn't have use it, but Davinici, but ...  21.5.2022)

def pix2pix(src_bw,dst_color,target_path,ext="jpg",ext_out="jpg",resize=None, start=0, maxn=9999999, cvt_src_to_bw=True): #8-2022; recreated 12-8-2022
   #top=100
   nm = start #index - to add to already created files
   if src_bw == None: print("Using the color input, converted to BW"); convert_src_to_grayscale = True; src_bw = dst_color; cvt_src_to_bw=True
   #or just copy the dir!... #29-8-2022
     
   print(ext_out)
   print(src_bw)
   print(dst_color)
   cv2.waitKey(0)
   #xx = input("gdsgdgfdgfd")
   print(os.system(f"dir {src_bw}"))
   print(os.system(f"dir {dst_color}"))
   s = glob.glob(src_bw+"*."+ext) #jpg") #or PNG etc.
   print(s)
   d = glob.glob(dst_color+"*."+ext) #+"*.jpg")
   print(d)
   #NO SORT etc. files.sort(reverse=True)
   
   #files.sort(reverse=True) #from the last ones
   #the last one repeat _last.jpg

   #import random
   #random.shuffle(files) #14-5-2022
   #!! diff.sizes
   aspect = 1.777777   
   w = 128 # 192 #per image
   h = 128 # 192
   # st = 192, 768
   if resize!=None:
     w,h = resize
   # eps = 0.01
   # w, h = 854, 480
   # out = []
   #maxn = 6
   n = 0
   #h_canvas, w_canvas = 720, 1280
   #h_canvas, w_canvas = 768, 1366
   
   h_canvas, w_canvas = w, 2*w
   
   #*7 = 1344
   #192 ... 
   #target_size = 6*192 = 1152
   #1280x720 ... target_size to fit 3x192 = 512
   #canvas = np.array(w*6, w*3 )   
   #canvas = np.array( (h_canvas, w_canvas), np.uint8)canvas =
   
   #canvas = np.zeros( (h_canvas, w_canvas), np.uint8)
   canvas = np.zeros( (h_canvas, w_canvas, 3),  np.uint8)
   #canvas = np.
   #canvas[:,:] = 0  - daa.. ne samo np.zeros(image.shape) etc. shape, np.float32, np.uint8 ....
   print(canvas.shape)
   
   for src_file,dst_file in zip(s,d):
     #canvas[:,:] = 0
     canvas[:] = 0
     print(canvas.shape)
     #cmd = f"rename {i} {i}_gamma.jpg"
     #cmd = f"ren {i} {i}_gamma.jpg"
     #src_img = cv2.imread(src_file,0)
     if cvt_src_to_bw:
       src_img = cv2.imread(src_file,0)         
     else: src_img = cv2.imread(src_file)
     print(src_img.shape)     
     #cv2.waitKey(0)
     if src_img.shape==2 or cvt_src_to_bw:
       src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
     #cv2.imshow("IMAGE", src_img)
     c1 = 1
     h1=0; w1=0
     if src_img.shape==3:
       h1, w1, c1 = src_img.shape
     #else:
     #  h1, w1 = src_img.shape       
     print(h1,w1,c1)
     dst_img = cv2.imread(dst_file) #,0) COLOR
     #cv2.imshow("IMAGE", dst_img)
     if resize != None:      
       src_img = cv2.resize(src_img, resize,cv2.INTER_LANCZOS4)       
       dst_img = cv2.resize(dst_img, resize, cv2.INTER_LANCZOS4)
     cv2.imshow("SRC_IMAGE", src_img)
     cv2.imshow("DST_IMAGE", dst_img)        
     #canvas[0:w,0:w] = src_img
     #canvas[0:w,w:w*2] = dst_img
     canvas[0:w,0:w] = dst_img
     canvas[0:w,w:w*2] = src_img
     cv2.imshow("CANV", canvas)
     cv2.waitKey(1)
     n+=1
     nm+=1
     if n > maxn: break      
     path_out = target_path+"-"+str(nm)+"."+ext_out
     print(path_out)
     #cv2.imwrite(target_path+"-"+str(nm)+"."+ext_out, canvas)   #finish                  
     cv2.imwrite(path_out, canvas)   #finish                  
     #finish                        
     
src="Z:\\faces\\"
target="Z:\\dataset\\"
pix2pix(src_bw=src, dst_color = src,target_path = target,ext="jpg",ext_out="jpg",resize=(128,128), start=1, maxn=9999999, cvt_src_to_bw=True) #30-8-2022


