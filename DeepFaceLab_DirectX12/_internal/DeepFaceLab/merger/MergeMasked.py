import sys
import traceback

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.interact import interact as io
from facelib import FaceType, LandmarksProcessor
import pathlib
import re #proper lexico sort of the files for the mapping, #20.8.2022

"""
ARNOLDIFIER experimental Colorization version of the merged
Extended by Todor Arnaudov, original base code by iperov
Twenkid's modified version of iperov's DeepfaceLab code
Sorry: highly experimental and not cleaned from commented code and tests
See also: 
1) merger\InteractiveMergerSubprocessor.py it has to be modified with the colorizing version as well.
2) ... .bat ... example of the workflow 

See also: ~ C:\DFL\DeepFaceLab_DirectX12\_internal\DeepFaceLab\merger\InteractiveMergerSubprocessor.py

"""
#from colorize import pix_128 #10-8-2022 --> call as a script

is_windows = sys.platform[0:3] == 'win'
xseg_input_size = 256

use_bw_input = "use_bw_input" in os.environ  #18-5-2022

debug_merge_masked = "debug_merge_masked" in os.environ #22-5-2022

reduce_colors_bw_number_steps = 16 #16 #23-5-2022 

#print(f"default_merge_cpu_count = {default_merge_cpu_count}")

counter = 0
apply_reshape_test = "apply_reshape_test" in os.environ #25-5-2022
#apply_reshape_test = True

not_inline = True

#Color frames should be parallel to the BW - check what calls here, draw graph etc. ...
colorize_pix2pix = "colorize_pix2pix" in os.environ 
has_color_frames_path = "has_color_frames_path" in os.environ #13-8-2022 -->
#9-8-2022
color_frames_path = None
if has_color_frames_path:
  color_frames_path = os.environ["color_frames_path"]

if "input_frames_path" in os.environ:
  input_frames_path = os.environ["input_frames_path"]
else: print("ERROR: Missing input_frames_path environment variable - required for multithread merging!")

print(f"has_color_frames_path={has_color_frames_path}")
print(f"color_frames_path={color_frames_path}")

# Colorization editing - up to 29.8.2022 ... then pause
# The sort: color_files_list.sort was because initially I didn't use a map, but a number, however it was not     enough,
# because DFL is a multiprocess program and the images were sent out of the sequential 
# order by a dispatcher process.
  
import glob
color_files_list = []
input_files_list = [] #14-8-2022
input_to_color_map = {} #14-8-2022

if has_color_frames_path: #adjust extension etc.! currently jpg
  color_files_list = glob.glob(color_frames_path+"*.jpg")
  print(f"glob.glob({color_frames_path}*.jpg")  
  print(color_files_list)  
  #color_files_list.sort()
  color_files_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]) 
  print(color_files_list)  
  print(f"dir {color_frames_path}")
  print(os.system(f"dir {color_frames_path}"))
  if input_frames_path != None: #needed to map frames in order to allow multithreading
    input_files_list = glob.glob(input_frames_path+"*.jpg")
    print(f"glob.glob({color_frames_path}.jpg")  
    print(input_files_list)  
    #color_files_list.sort()
    input_files_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]) 
    print(input_files_list)  
    print(f"dir {input_frames_path}")
    print(os.system(f"dir {input_frames_path}"))
    for i,c in zip(input_files_list, color_files_list):
      #input_to_color_map[i] = c    
      input_to_color_map[pathlib.Path(i).name] = c # no, color file is complete path! pathlib.Path(c).name
    #T:\\stolt-bw-1144-1294\\ 
#Differences in \\ / etc. --> either normalize or use only the file name    
  
else: print("Doesn't have has_color_frames_path?")

ss = input("Pause to see has_color_frames_path")

count_color_frame = 0 #make better, mapping etc.  --> in Merge ... cunt
#counted separately from separate threads? use frame_info? -- not of course
#Make a map or add to frame_info #14-8-2022

only_colorize_dont_predict="only_colorize_dont_predict" in os.environ #13-8-2022
show_out_img = False
color_file = "" #14-8-2022

use_precomputed_faces = "use_precomputed_faces" in os.environ #19-8-2022
if use_precomputed_faces: precomputed_faces_path = os.environ["precomputed_faces_path"]

precomputed_faces_list = []
input_to_precomputed_map = {}

if use_precomputed_faces: #adjust extension etc.! currently jpg
  print("use_precomputed_faces")
  #precomputed_faces_list = glob.glob(precomputed_faces_path+"*.jpg") #set extension etc.!
  print(f"{precomputed_faces_path}*.*")
  precomputed_faces_list = glob.glob(precomputed_faces_path+"*.*") #set extension etc.!
  #print(f"glob.glob({precomputed_faces_path}.jpg")  
  print(precomputed_faces_list)  
  #precomputed_faces_list.sort()
  precomputed_faces_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]) 
  print(precomputed_faces_path)  
  print(f"dir {precomputed_faces_path}")
  print(os.system(f"dir {precomputed_faces_path}"))
  if input_frames_path != None: #needed to map frames in order to allow multithreading
    #input_files_list = glob.glob(input_frames_path+"*.jpg") # ADJUST EXT., include PNG etc.!
    input_files_list = glob.glob(input_frames_path+"*.*") # ADJUST EXT., include PNG etc.!
    #print(f"glob.glob({input_frames_path}.jpg")  
    print(f"glob.glob({input_frames_path}*.*")  
    print(input_files_list)  
    #input_files_list.sort()
    input_files_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    print(input_files_list)  
    print(f"dir {precomputed_faces_list}")
    print(os.system(f"dir {precomputed_faces_list}"))
    for i,c in zip(input_files_list, precomputed_faces_list):
      #input_to_color_map[i] = c    
      input_to_precomputed_map[pathlib.Path(i).name] = c # no, color file is complete path! pathlib.Path(c).name
      print(f"###{pathlib.Path(i).name}-->{c}")
    #T:\\stolt-bw-1144-1294\\

#colorize_path = os.environ[...
#Careful if input is color 
def MergeMaskedFace (predictor_func, predictor_input_shape,
                     face_enhancer_func,
                     xseg_256_extract_func,
                     cfg, frame_info, img_bgr_uint8, img_bgr, img_face_landmarks):
    global count_color_frame
    global color_files_list
    
    if debug_merge_masked: print("MergeMaskedFace")
    if debug_merge_masked: print(f"img_bgr.shape={img_bgr.shape}")
    img_size = img_bgr.shape[1], img_bgr.shape[0]
    img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)
    print(frame_info)   
    img_bgr_color = None #if not colorizing
    if colorize_pix2pix and has_color_frames_path: #13-8-2022
      print("Current frame path: ", frame_info.filepath)
      #print("Color frame path: ", color_files_list[count_color_frame])
      print("input_to_color_map", input_to_color_map)
      print("Color frame path: ", input_to_color_map[frame_info.filepath.name])
      print(f"input_frame to color_frame map: \n{frame_info.filepath}\n{input_to_color_map[frame_info.filepath.name]}") #the filename is the key
      #print(f"filename only: input_frame to color_frame map: \n{frame_info.filepath.name}\n{input_to_color_map[frame_info.filepath]}")
            
      #color_file = input_to_color_map[frame_info.filepath]
      color_file = input_to_color_map[frame_info.filepath.name]      
      
      #color_filepath = frame_info.filepath            
      ho = img_bgr.shape[0]
      wo = img_bgr.shape[1]
      aspect_original = wo/ho #img_bgr.shape[1] / img_bgr.shape[0]      
      #img_bgr = cv2.imread(color_files_list[count_color_frame])
      
      #img_bgr_uint8 = cv2.imread(color_files_list[count_color_frame])
      
      
      #img_bgr_color_uint8 = cv2.imread(color_files_list[count_color_frame]) ### NO - READ THAT LATER? OR shouldn't be a problem - the face is read from the img_bgr to 13-8-2022
      img_bgr_color_uint8 = cv2.imread(color_file)  #from map
      
      #hc, wc, cc = img_bgr.shape
      hc = img_bgr.shape[0]
      wc = img_bgr.shape[1]
      aspect_color = wc/hc #img_bgr.shape[1] / img_bgr.shape[0]
      #currently has to resize      
      #print("ASPECTS: original, 
      scale = wo/wc #1280/1920
      print(f"ASPECTs and scale  {aspect_original}, {aspect_color}, {scale}")
      #if aspect_color!= aspect_original:
      if scale < 1:
        #should compute once etc. powers of two/divisible ... 
        #for now fixed
        #img_bgr = cv2.resize(int(scale* ...
        print("Resizing to 1280x720")
        #img_bgr = cv2.resize(img_bgr, (1280,720), cv2.INTER_CUBIC) #cv2.INTER_LANCZOS4) #SLOW BUT HIGHEST QUALITY - maybe not needed?           
        #img_bgr_uint8 = cv2.resize(img_bgr, (1280,720), cv2.INTER_CUBIC) #cv2.INTER_LANCZOS4) #SLOW BUT HIGHEST QUALITY - maybe not needed?           
        img_bgr_color_uint8 = cv2.resize(img_bgr_color, (1280,720), cv2.INTER_CUBIC) #cv2.INTER_LANCZOS4) #SLOW BUT HIGHEST QUALITY - maybe not needed?           
        #img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        #img_bgr = img_bgr_uint8/255 #cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
      #img_bgr_color = img_bgr_uint8/255 #cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR) ERROR -- must be _color!
      img_bgr_color = img_bgr_color_uint8/255 #cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR) ERROR -- must be
        #img_bgr = img_bgr / 255 #cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
      count_color_frame+=1
            
    input_size = predictor_input_shape[0]
    mask_subres_size = input_size*4
    output_size = input_size
    if cfg.super_resolution_power != 0:
        output_size *= 4

    face_mat        = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=cfg.face_type)
    face_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=cfg.face_type, scale= 1.0 + 0.01*cfg.output_face_scale)

    if mask_subres_size == output_size:
        face_mask_output_mat = face_output_mat
    else:
        face_mask_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, mask_subres_size, face_type=cfg.face_type, scale= 1.0 + 0.01*cfg.output_face_scale)

    if debug_merge_masked:
      print("BEFORE: dst_face_bgr = cv2.warpAffine...")
      print(f"img_bgr.shape={img_bgr.shape}, face_mat.shape={face_mat.shape}, output_size={output_size}")
    
    #if use_bw_input and img_bgr.shape==2 and not colorize_pix2pix: #if color -- it will be 3; and not colorize_pix2pix -- #10-8-2022
    if use_bw_input and img_bgr.shape==2: # and not colorize_pix2pix: #if color -- it will be 3; and not colorize_pix2pix -- #10-8-2022
      print("if use_bw_input and img_bgr.shape==2:")
      img_bgr = img_bgr[...,np.newaxis]
    print(f"img_bgr={img_bgr}")
    print(f"img_bgr_color={img_bgr_color}")
    #if user_bw_input: 
    #    dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    #cv2.INTER_LANCZOS4
    
    dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    print(f"dst_face_bgr={dst_face_bgr}")
    
    #cv2.imshow("dst_face_bgr", dst_face_bgr)
    
    if debug_merge_masked: print(f"AFTER: {dst_face_bgr.shape}")
    
    dst_face_bgr      = np.clip(dst_face_bgr, 0, 1) 
    
    dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

    if debug_merge_masked: print("BEFORE: {predictor_input_bgr = cv2.resize...}")
        
    #if not only_colorize_dont_predict: #if ... #13-8-2022
    predictor_input_bgr      = cv2.resize (dst_face_bgr, (input_size,input_size) )
    if debug_merge_masked: print(f"AFTER: predictor_input_bgr.shape=predictor_input_bgr.shape={predictor_input_bgr.shape}, dst_face_bgr.shape={dst_face_bgr.shape}, input_size={input_size}")
    if debug_merge_masked: print("BEFORE:predicted = predictor_func (predictor_input_bgr)")
    
    #CAREFUL if color image and predicting from a grayscale model    
    if use_bw_input and colorize_pix2pix: #10-8-2022
      print(f"Shape of predictor_input_bgr before cvtColor BGR2GRAY = {predictor_input_bgr.shape}")
      print(predictor_input_bgr)
      #predictor_input_bgr = cv2.cvtColor(predictor_input_bgr.transpose(1,2,0), cv2.COLOR_BGR2GRAY)
      predictor_input_bgr = cv2.cvtColor(predictor_input_bgr, cv2.COLOR_BGR2GRAY)   
      print(f"Shape of predictor_input_bgr AFTER cvtColor BGR2GRAY = {predictor_input_bgr.shape}")
      
    #else: if use_precomputed_faces:            
    predicted = predictor_func (predictor_input_bgr)
    
    #"""
    #! that should be after? -- #19-8-2022 - error
    if only_colorize_dont_predict: 
      prd_face_bgr = np.clip(dst_face_bgr, 0, 1.0)
      print(f"if only_colorize_dont_predict:\nprd_face_bgr.shape={prd_face_bgr.shape}")
    #print(
    #13-8-2022 Takes it from the frame, not from the NN. predictor_func is alwayscalled in order to get the other values
    #"""
    else:
      prd_face_bgr          = np.clip (predicted[0], 0, 1.0) #BW
    prd_face_mask_a_0     = np.clip (predicted[1], 0, 1.0)
    prd_face_dst_mask_a_0 = np.clip (predicted[2], 0, 1.0)
    #! that should be after?
    ###if only_colorize_dont_predict: prd_face_bgr = np.clip(dst_face_bgr, 0, 1.0)
    #13-8-2022 Takes it from the frame, not from the NN. predictor_func is alwayscalled in order to get the other values               
    
    ### Colorizing with pix2pix model #9-8-2022
        
    print(f"#9-8-2022: 31-7-2022: MergeMasked: predicted.shape={prd_face_bgr.shape}")    

    prd_show = prd_face_bgr.copy()
    if prd_show.shape[2] > 5: 
      prd_show = prd_show.transpose(1,2,0) #.reshape( t.shape[-1],-1) ##comme.#19-8-2022
      
    print(f"prd_show.transpose(1,2,0) shape={prd_face_bgr.shape}")
    #_mask_a_0 --> alpha?
    #Check the shape etc., is there a dst to dst? (the same face) available from the predictor_func? How it is produced in the preview etc., S, SS, ... 
    cv2.imshow("PRD_FACE_BGR", prd_show) #prd_face_bgr)
    cv2.waitKey(1)
    #cv2.waitKey()
    #cv2.imwrite(prd_show, "T:\\prd.png")     #OK!
    #Write after color correction sot-m
    #cv2.imwrite("T:\\input_image.png",prd_show*255.0)  #OK!
    
    
    """
    #BUT NOT HERE --> after color transfer
    #Call a script to convert it    
    os.system("python  ...") #--save to file (another version of Python)
    #load the colorized image etc. --> but if colorizing, add another flag/ignore use_bw etc.!
    ###
    """
  
    if cfg.super_resolution_power != 0: #only if !=0
        """
        if use_bw_input: #etc. #13-8-2022
          #cv2.cvtColor(prd_face_bgr, cv2.COLOR_GRAY2BGR)
          prd_transposed = prd_face_bgr.transpose(1,2,0)
          prd_face_bgr = cv2.merge( (prd_transposed,prd_transposed,prd_transposed) )          
          prd_face_bgr = prd_face_bgr.transpose(2,0,1) #? back
        """                         
        prd_face_bgr_enhanced = face_enhancer_func(prd_face_bgr, is_tanh=True, preserve_size=False)
        mod = cfg.super_resolution_power / 100.0
        prd_face_bgr = cv2.resize(prd_face_bgr, (output_size,output_size))*(1.0-mod) + prd_face_bgr_enhanced*mod
        prd_face_bgr = np.clip(prd_face_bgr, 0, 1)
        if use_bw_input: #etc. #etc. #13-8-2022
           prd_face_bgr, _, _ = cv2.split(prd_face_bgr)

    if cfg.super_resolution_power != 0:
        prd_face_mask_a_0     = cv2.resize (prd_face_mask_a_0,      (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        prd_face_dst_mask_a_0 = cv2.resize (prd_face_dst_mask_a_0,  (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    if cfg.mask_mode == 0: #full
        wrk_face_mask_a_0 = np.ones_like(dst_face_mask_a_0)
    elif cfg.mask_mode == 1: #dst
        wrk_face_mask_a_0 = cv2.resize (dst_face_mask_a_0, (output_size,output_size), interpolation=cv2.INTER_CUBIC)
    elif cfg.mask_mode == 2: #learned-prd
        wrk_face_mask_a_0 = prd_face_mask_a_0
    elif cfg.mask_mode == 3: #learned-dst
        wrk_face_mask_a_0 = prd_face_dst_mask_a_0
    elif cfg.mask_mode == 4: #learned-prd*learned-dst
        wrk_face_mask_a_0 = prd_face_mask_a_0*prd_face_dst_mask_a_0
    elif cfg.mask_mode == 5: #learned-prd+learned-dst
        wrk_face_mask_a_0 = np.clip( prd_face_mask_a_0+prd_face_dst_mask_a_0, 0, 1)
    elif cfg.mask_mode >= 6 and cfg.mask_mode <= 9:  #XSeg modes
        if cfg.mask_mode == 6 or cfg.mask_mode == 8 or cfg.mask_mode == 9:
            # obtain XSeg-prd
            prd_face_xseg_bgr = cv2.resize (prd_face_bgr, (xseg_input_size,)*2, interpolation=cv2.INTER_CUBIC)
            prd_face_xseg_mask = xseg_256_extract_func(prd_face_xseg_bgr)
            X_prd_face_mask_a_0 = cv2.resize ( prd_face_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

        if cfg.mask_mode >= 7 and cfg.mask_mode <= 9:
            # obtain XSeg-dst
            xseg_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, xseg_input_size, face_type=cfg.face_type)
            dst_face_xseg_bgr   = cv2.warpAffine(img_bgr, xseg_mat, (xseg_input_size,)*2, flags=cv2.INTER_CUBIC )
            dst_face_xseg_mask  = xseg_256_extract_func(dst_face_xseg_bgr)
            X_dst_face_mask_a_0 = cv2.resize (dst_face_xseg_mask, (output_size,output_size), interpolation=cv2.INTER_CUBIC)

        if cfg.mask_mode == 6:   #'XSeg-prd'
            wrk_face_mask_a_0 = X_prd_face_mask_a_0
        elif cfg.mask_mode == 7: #'XSeg-dst'
            wrk_face_mask_a_0 = X_dst_face_mask_a_0
        elif cfg.mask_mode == 8: #'XSeg-prd*XSeg-dst'
            wrk_face_mask_a_0 = X_prd_face_mask_a_0 * X_dst_face_mask_a_0
        elif cfg.mask_mode == 9: #learned-prd*learned-dst*XSeg-prd*XSeg-dst
            wrk_face_mask_a_0 = prd_face_mask_a_0 * prd_face_dst_mask_a_0 * X_prd_face_mask_a_0 * X_dst_face_mask_a_0

    wrk_face_mask_a_0[ wrk_face_mask_a_0 < (1.0/255.0) ] = 0.0 # get rid of noise

    # resize to mask_subres_size
    if wrk_face_mask_a_0.shape[0] != mask_subres_size:
        wrk_face_mask_a_0 = cv2.resize (wrk_face_mask_a_0, (mask_subres_size, mask_subres_size), interpolation=cv2.INTER_CUBIC)

    if debug_merge_masked: print("if raw not in cfg.mode...")
    # process mask in local predicted space
    if 'raw' not in cfg.mode:
        if debug_merge_masked: print("if raw not in cfg.mode...")
        # add zero pad
        wrk_face_mask_a_0 = np.pad (wrk_face_mask_a_0, input_size)

        ero  = cfg.erode_mask_modifier
        blur = cfg.blur_mask_modifier

        if ero > 0:
            wrk_face_mask_a_0 = cv2.erode(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
        elif ero < 0:
            wrk_face_mask_a_0 = cv2.dilate(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )

        # clip eroded/dilated mask in actual predict area
        # pad with half blur size in order to accuratelly fade to zero at the boundary
        clip_size = input_size + blur // 2

        wrk_face_mask_a_0[:clip_size,:] = 0
        wrk_face_mask_a_0[-clip_size:,:] = 0
        wrk_face_mask_a_0[:,:clip_size] = 0
        wrk_face_mask_a_0[:,-clip_size:] = 0

        if blur > 0:
            blur = blur + (1-blur % 2)
            wrk_face_mask_a_0 = cv2.GaussianBlur(wrk_face_mask_a_0, (blur, blur) , 0)

        wrk_face_mask_a_0 = wrk_face_mask_a_0[input_size:-input_size,input_size:-input_size]

        wrk_face_mask_a_0 = np.clip(wrk_face_mask_a_0, 0, 1)

    #if not use_bw_input:
        if debug_merge_masked: print(f"\nMergeMaskedFace:img_bgr.shape={img_bgr.shape}")
    img_face_mask_a = cv2.warpAffine( wrk_face_mask_a_0, face_mask_output_mat, img_size, np.zeros(img_bgr.shape[0:2], dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )[...,None]
    #else: #18-8-2022
    #     img_face_mask_a = cv2.warpAffine( wrk_face_mask_a_0, face_mask_output_mat, img_size, np.zeros(img_bgr.shape[0:2], dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )[...,None]
             
    img_face_mask_a = np.clip (img_face_mask_a, 0.0, 1.0)
    img_face_mask_a [ img_face_mask_a < (1.0/255.0) ] = 0.0 # get rid of noise

    if wrk_face_mask_a_0.shape[0] != output_size:
        wrk_face_mask_a_0 = cv2.resize (wrk_face_mask_a_0, (output_size,output_size), interpolation=cv2.INTER_CUBIC)

    wrk_face_mask_a = wrk_face_mask_a_0[...,None]

    out_img = None
    out_merging_mask_a = None   
    
    if debug_merge_masked: print(" if cfg.mode == 'original'...")
    if cfg.mode == 'original':
        return img_bgr, img_face_mask_a
    elif 'raw' in cfg.mode:
        if cfg.mode == 'raw-rgb':
            if debug_merge_masked:  print(f"prd_face_bgr.shape={prd_face_bgr.shape}")
            
            out_img_face = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
            out_img_face_mask = cv2.warpAffine( np.ones_like(prd_face_bgr), face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
            out_img = img_bgr*(1-out_img_face_mask) + out_img_face*out_img_face_mask
            out_merging_mask_a = img_face_mask_a
        elif cfg.mode == 'raw-predict':
            out_img = prd_face_bgr
            out_merging_mask_a = wrk_face_mask_a
        else:
            raise ValueError(f"undefined raw type {cfg.mode}")

        out_img = np.clip (out_img, 0.0, 1.0 )
    else:

        # Process if the mask meets minimum size
        maxregion = np.argwhere( img_face_mask_a >= 0.1 )
        if maxregion.size != 0:
            miny,minx = maxregion.min(axis=0)[:2]
            maxy,maxx = maxregion.max(axis=0)[:2]
            lenx = maxx - minx
            leny = maxy - miny
            if min(lenx,leny) >= 4:
                wrk_face_mask_area_a = wrk_face_mask_a.copy()
                wrk_face_mask_area_a[wrk_face_mask_area_a>0] = 1.0

                if 'seamless' not in cfg.mode and cfg.color_transfer_mode != 0:
                    if cfg.color_transfer_mode == 1: #rct
                        prd_face_bgr = imagelib.reinhard_color_transfer (prd_face_bgr, dst_face_bgr, target_mask=wrk_face_mask_area_a, source_mask=wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 2: #lct
                        if not_inline:  
                                print(f"LCT cfg.color_transfer_mode == 2 prd_face_bgr.shape= {prd_face_bgr.shape},dst_face_bgr.shape={dst_face_bgr.shape}")
                                if prd_face_bgr.shape[0] ==1:
                                    prd_face_bgr = prd_face_bgr.transpose(1,2,0)
                                    print(f"prd_face_bgr.transpose(1,2,0), prd_face_bgr.shape= {prd_face_bgr.shape}")
                                if len(dst_face_bgr.shape) ==2:
                                    dst_face_bgr = dst_face_bgr[...,np.newaxis]                          
                                    #dst_face_bgr = dst_face_bgr.transpose(1,2,0)
                                    print(f"dst_face_bgr.shape, {dst_face_bgr.shape}")
                            
                                prd_face_bgr = imagelib.linear_color_transfer (prd_face_bgr, dst_face_bgr)
                            
                        else:
                        
                            print("elif cfg.color_transfer_mode == 2: #lct")
                            
                            print(f"LCT cfg.color_transfer_mode == 7 prd_face_bgr.shape= {prd_face_bgr.shape},dst_face_bgr.shape={dst_face_bgr.shape}")
                            if prd_face_bgr.shape[0] ==1:
                              prd_face_bgr = prd_face_bgr.transpose(1,2,0)
                              print(f"prd_face_bgr.transpose(1,2,0), prd_face_bgr.shape= {prd_face_bgr.shape}")
                            if len(dst_face_bgr.shape) == 2:
                              dst_face_bgr = dst_face_bgr[...,np.newaxis]                          
                              #dst_face_bgr = dst_face_bgr.transpose(1,2,0)
                            print(f"dst_face_bgr.shape, {dst_face_bgr.shape}")
                            
                            
                            
                            
                            #target_img, source_img, mode='pca', eps=1e-5):
                            
                            ######### INLINE AND SEE... ####
                            '''
                            Matches the colour distribution of the target image to that of the source image
                            using a linear transform.
                            Images are expected to be of form (w,h,c) and float in [0,1].
                            Modes are chol, pca or sym for different choices of basis.
                            '''
                            mode='pca'; eps=1e-5
                            target_img = prd_face_bgr.copy()
                            source_img = dst_face_bgr.copy()
                            print("INLINED: linear_color_transfer! 26-5-2022")
                            global counter
                            print(f"Call#{counter}"); counter+=1  #it's called only once??? #26-5-2022
                            print(f"apply_reshape_test = {apply_reshape_test}")
                            print(f"color_transfer.py: linear_color_transfer,target_img.shape={target_img.shape}, source_img.shape={source_img.shape}")
                            
                               
                            #color_transfer.py: linear_color_transfer,target_img.shape=(1, 192, 192), source_img.shape=(192, 192)
                            if use_bw_input: #18-5-2022
                              if len(target_img.shape) == 3:
                                s1, s2, s3 = target_img.shape
                              else:
                                s1, s2 = target_img.shape
                                
                              if s1 == 1: 
                                #target_img = np.squeeze(target_img); target_img = target_img[...,np.newaxis] # for mergings #was up to #26-5-2022, but wrong
                                #1,192,192 --> should be 192,192,1; will it fix it?
                                #target_img = target_img.transpose(2,0,1) # for mergings #was up to #26-5-2022, but wrong
                                target_img = target_img.transpose(1,2,0) # for mergings #was up to #26-5-2022, but wrong
                                print(f"TRANSPOSE (1,2,0) (1,192,192-->192,192,1?): target_img.shape={target_img.shape}, np.sum(target_img)={np.sum(target_img)}")
                                print(f"np.sum(source_img)={np.sum(source_img)}")
                              L1 = len(source_img.shape)
                              if L1 == 2: source_img = source_img[..., np.newaxis]
                              print(f"color_transfer.py: linear_color_transfer,target_img.shape={target_img.shape}, source_img.shape={source_img.shape}")
                            #cv2.waitKey(0)
                           
                            #if len(target_img.shape)==2: target_img = target_img[:,None] #np.newaxis #18-5-2022 
                            mu_t = target_img.mean(0).mean(0)
                            t = target_img - mu_t    
                            
                            if not use_bw_input: #24-5-2022 --> that must be the colors, BGR to RGB
                              t = t.transpose(2,0,1).reshape( t.shape[-1],-1)
                            else:      
                              if apply_reshape_test:         
                                  t = t.reshape( t.shape[-1],-1) #?24-5-2022  --> no transpose, but reshape? or don't?
                                  if t.shape[0] == 1: s = np.squeeze(t); 
                                  print(f"t = np.squeeze(t); t.shape = {t.shape}")
                              
                              print(f"t.reshape(t.shape[-1],-1)={t.shape}")
                              #TRY TO RETURN HERE, #26-5-2022
                              #return np.clip(target_img.astype(source_img.dtype), 0, 1)
                              #WHEN RETURNING HERE --> on_result in ... Base is reached
                            print(f"t.dot, t.shape[1], np.eye[t.shape[0= = {t.dot(t.T)}, {t.shape[1]}, {np.eye(t.shape[0])}")
                            #
                            Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
                            print("After Ct = ...")
                            mu_s = source_img.mean(0).mean(0)
                            print("After mu_s = ...")
                            s = source_img - mu_s
                            print("After s = source_img - mu_s = ...")    
                            #OK, reaches here
                            #return np.clip(target_img.astype(source_img.dtype), 0, 1)
                            
                            
                            if not use_bw_input:
                              s = s.transpose(2,0,1).reshape( s.shape[-1],-1)
                            else:
                              if apply_reshape_test:
                                  print(f"apply_reshape_test: s.shape={s.shape}")
                                  s = s.reshape( s.shape[-1],-1) #?24-5-2022  --> no transpose, but reshape? or don't?
                                  print(f"s.reshape(t.shape[-1],-1)={s.shape}")  
                                  if s.shape[0] == 1: s = np.squeeze(s); print(f"s = np.squeeze(s); s.shape = {s.shape}")
                                  s = s.reshape( s.shape[-1],-1)  #24-5-2022
                                  print(f"s = s.reshape(....) {s.shape}")         
                            print(f"Before Cs = ... KURRRRRRRRRRRRRR?!!!!")
                            print(f"\nCs= s.dot(s.T) = {s.dot(s.T)}, ... {s.shape[1]}, {np.eye(t.shape[0])}") 
                            prd_face_bgr =  np.clip(target_img.astype(target_img.dtype), 0, 1) #FORCED
                            Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
                            print(f"np.sum(Cs)={np.sum(Cs)}")
                            #return np.clip(target_img.astype(source_img.dtype), 0, 1)
                            if mode == 'chol':
                                chol_t = np.linalg.cholesky(Ct)
                                chol_s = np.linalg.cholesky(Cs)
                                ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)        
                            if mode == 'pca':
                                print("mode=pca")
                                eva_t, eve_t = np.linalg.eigh(Ct)
                                Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
                                eva_s, eve_s = np.linalg.eigh(Cs)
                                Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
                                ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
                            if mode == 'sym':
                                eva_t, eve_t = np.linalg.eigh(Ct)
                                Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
                                Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
                                eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
                                QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
                                ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
                            print(f"ts={ts}")
                            if not use_bw_input: 
                              matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
                            else: #matched_img = ts    
                                  #matched_img = ts #ERROR!
                                  matched_img = ts.reshape(target_img.shape) #27-5-2022  
                            print(f"np.sum(matched_mg)={np.sum(matched_img)}")
                            #if use_bw_input: 
                            print(f"matched_img.shape={matched_img.shape}")
                            matched_img += mu_s
                            matched_img[matched_img>1] = 1
                            matched_img[matched_img<0] = 0
                            print(f"matched_img.shape. sum(matched_img){matched_img.shape}, {np.sum(matched_img)}")
                            
                            prd_face_bgr =  np.clip(matched_img.astype(source_img.dtype), 0, 1)
                            #return np.clip(matched_img.astype(source_img.dtype), 0, 1)
                                                    
                            
                            """
                            try:
                              prd_face_bgr = imagelib.linear_color_transfer (prd_face_bgr, dst_face_bgr)
                            except:
                                  print("Exception in linear_color_transfer!!!"+sys.exc_info());
                            """
                            #It doesn't reach here. WHY?! No exception, just "disappears"before Cs = ... in linear_color_transfer #26-5-2022 
                            print(f"np.sum(prd_face_bgr)={np.sum(prd_face_bgr)}")
                    elif cfg.color_transfer_mode == 3: #mkl
                        prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 4: #mkl-m
                        prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 5: #idt
                        prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 6: #idt-m
                        prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 7: #sot-m
                        print(f"SOT cfg.color_transfer_mode == 7 prd_face_bgr.shape= {prd_face_bgr.shape},dst_face_bgr.shape={dst_face_bgr.shape}")
                        if prd_face_bgr.shape[0] ==1:
                          prd_face_bgr = prd_face_bgr.transpose(1,2,0)
                          print(f"prd_face_bgr.transpose(1,2,0), prd_face_bgr.shape= {prd_face_bgr.shape}")
                        if len(dst_face_bgr.shape) ==2:
                          dst_face_bgr = dst_face_bgr[...,np.newaxis]                          
                          #dst_face_bgr = dst_face_bgr.transpose(1,2,0)
                        print(f"dst_face_bgr.shape, {dst_face_bgr.shape}")                          
                        
                        prd_face_bgr = imagelib.color_transfer_sot (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a, steps=10, batch_size=30)
                        prd_face_bgr = np.clip (prd_face_bgr, 0.0, 1.0)
                        #sot-m color transferred, 0.0,1.0
                        
                    elif cfg.color_transfer_mode == 8: #mix-m
                        prd_face_bgr = imagelib.color_transfer_mix (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)

                if cfg.mode == 'hist-match':
                  if not use_bw_input:
                    hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                    if cfg.masked_hist_match:
                        hist_mask_a *= wrk_face_mask_area_a

                    white =  (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                    hist_match_1 = prd_face_bgr*hist_mask_a + white
                    hist_match_1[ hist_match_1 > 1.0 ] = 1.0

                    hist_match_2 = dst_face_bgr*hist_mask_a + white
                    hist_match_2[ hist_match_1 > 1.0 ] = 1.0

                    prd_face_bgr = imagelib.color_hist_match(hist_match_1, hist_match_2, cfg.hist_match_threshold ).astype(dtype=np.float32)
                  else: #BW, grayscale, #22-5-2022
                    #hist_mask_a = np.ones ( prd_face_bgr.shape[0:] + (1,) , dtype=np.float32)
                    #hist_mask_a = np.ones ( prd_face_bgr.shape[0:2] + (1,) , dtype=np.float32)
                    #shape: 1,1,192,192
                    #hist_mask_a = np.ones ( prd_face_bgr.shape[2:3] + (1,) , dtype=np.float32)
                    if debug_merge_masked: print(f"prd_face_bgr.shape={prd_face_bgr.shape}")
                    #hist_mask_a = np.ones ( prd_face_bgr.shape[0:3] + (1,) , dtype=np.float32)
                    hist_mask_a = np.ones ( (192,192) + (1,) , dtype=np.float32) #FORCED
                    #or just prd_face_bgr.shape)
                    if debug_merge_masked:
                      print(f"hist_mask_a.shape={hist_mask_a.shape}")
                      print(f"wrk_face_mask_area_a.shape={wrk_face_mask_area_a.shape}")
                      print(f"dst_face_bgr.shape={dst_face_bgr.shape}")
                    if cfg.masked_hist_match:
                        hist_mask_a *= wrk_face_mask_area_a

                    #white =  (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[0:2] + (1,) , dtype=np.float32)
                    white =  (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape, dtype=np.float32)
                    if debug_merge_masked:
                      print("white.shape = {white.shape}")
                    
                    hist_match_1 = prd_face_bgr*hist_mask_a + white
                    hist_match_1[ hist_match_1 > 1.0 ] = 1.0
                    if debug_merge_masked:
                      print(f"hist_match_1.shape={hist_match_1.shape}")
                    if len(dst_face_bgr.shape)==2:
                      dst_face_bgr = dst_face_bgr[:,:,None] #new axis ? 
                      if debug_merge_masked:
                        print(f"if len(dst_face_bgr.shape==2): --> add axis, dst_face_bgr.shape={dst_face_bgr.shape}")
                    hist_match_2 = dst_face_bgr*hist_mask_a + white
                    hist_match_2[ hist_match_1 > 1.0 ] = 1.0

                    if debug_merge_masked: 
                      print(f"hist_match_2.shape={hist_match_2.shape}")
                    
                    #prd_face_bgr = imagelib.color_hist_match(hist_match_1, hist_match_2, cfg.hist_match_threshold ).astype(dtype=np.float32)
                    prd_face_bgr = imagelib.bw_hist_match(hist_match_1, hist_match_2, cfg.hist_match_threshold ).astype(dtype=np.float32)
                    
                       

                if 'seamless' in cfg.mode:
                    #mask used for cv2.seamlessClone
                    img_face_seamless_mask_a = None
                    for i in range(1,10):
                        a = img_face_mask_a > i / 10.0
                        if len(np.argwhere(a)) == 0:
                            continue
                        img_face_seamless_mask_a = img_face_mask_a.copy()
                        img_face_seamless_mask_a[a] = 1.0
                        img_face_seamless_mask_a[img_face_seamless_mask_a <= i / 10.0] = 0.0
                        break
                
                if debug_merge_masked: print(f"BEFORE out_img = ... {prd_face_bgr.shape}, {face_output_mat.shape}, {img_size}, {img_bgr.shape}, ")
                
                
                ### prd_face_bgr!!! That is to colorize! #10-8-2022
                #CALL THIS IN MergeMaskeface?
                #COLORIZE
                #cv2.imwrite("T:\\input_image.png",prd_show*255.0)  #OK!
                
                """
                prd_show = prd_face_bgr.copy()
                if prd_face_show.shape[2] > 4: 
                  prd_show = prd_show.transpose(1,2,0)                   
                print(f"prd_face_bgr={prd_face_bgr.shape}")
                print(f"prd_show={prd_show.shape}")
                #cv2.imwrite("T:\\input_image.png",prd_face_bgr*255.0)  #OK!                
                #cv2.imwrite("T:\\input_image.png ",prd_show*255.0)  #OK! so far 13-8-2022
                """
                png_compression = 0 # fastest
                #cv2.imwrite(f"T:\\input_image_{count_color_frame}.png",prd_show*255.0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])  #14-8-2022 --> run several threads
                
                if only_colorize_dont_predict:
                  prd_show_bw = cv2.cvtColor(prd_show, cv2.COLOR_BGR2GRAY)
                  
                  cv2.imwrite(f"Z:\\input_image_{frame_info.filepath.name}.png",prd_show_bw*255.0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])  #14-8-2022 --> run several threads
                  print(f"only_colorize_dont_predict:\nframe_info.filepath.name={frame_info.filepath.name}")
                #19-8-2022                 
                else:
                  cv2.imwrite(f"Z:\\input_image_{frame_info.filepath.name}.png",prd_show*255.0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])  #14-8-2022 --> run several threads
                  print(f"frame_info.filepath.name={frame_info.filepath.name}")
                
                #input_image = sys.argv[0]
                #predicted_image = sys.arg[1]
                
                
                #cv2.imwrite("T:\\input_image.bmp",prd_show*255.0)  #OK! try bmp
                
                if colorize_pix2pix and not use_precomputed_faces:
                    
                    #os.system(f"python C:\\DFL\\DeepFaceLab_DirectX12\\_internal\\DeepFaceLab\\colorize\\pix_128.py T:\\input_image_{count_color_frame}.png T:\\predicted_{count_color_frame}.png")
                    print(f"if colorize_pix2pix: {frame_info.filepath.name}.png, {frame_info.filepath.name}.png")
                    os.system(f"python C:\\DFL\\DeepFaceLab_DirectX12\\_internal\\DeepFaceLab\\colorize\\pix_128.py Z:\\input_image_{frame_info.filepath.name}.png Z:\\predicted_{frame_info.filepath.name}.png")
                    #out_img = cv2.imread("T:\\predicted.png") #_image.png")
                    #out_img_face = cv2.imread("T:\\predicted.png") #_image.png")                                       
                    
                    #prd_face_bgr = cv2.imread("T:\\predicted.png") #_image.png")   so far #13-8-2022
                    #prd_face_bgr = cv2.imread("T:\\predicted_{count_color_frame}.png") #  so far #13-8-2022                   
                    print(" Z:\\predicted_{frame_info.filepath.name}.png")
                    prd_face_bgr = cv2.imread(f"Z:\\predicted_{frame_info.filepath.name}.png") #  14-8-2022 far #13-8-2022                                      
                    #RESIZE TO THE PROPER RESOLUTION!
                    prd_face_bgr = cv2.resize(prd_face_bgr, (192, 192), interpolation=cv2.INTER_CUBIC)
                    #prd_face_bgr = cv2.imread("T:\\predicted.bmp") #_image.png")                    
                    cv2.imshow("PREDICTED IN DFL", prd_face_bgr)
                    prd_face_bgr = prd_face_bgr/ 255.0 #?
                    cv2.waitKey(1)
                #pix2pix now is only 128x128, model is 192
                #out_img_face = cv2.resize(prd_face_bgr, (192, 192), interpolation=cv2.INTER_CUBIC)
                #
                else:
                  if use_precomputed_faces: #20-8-2022
                      #T:\adjusted
                      #only read 
                      precomputed_face = input_to_precomputed_map[frame_info.filepath.name]
                      
                      print(f"input_to_precomputed_map[frame_info.filepath.name]-->\n{input_to_precomputed_map[frame_info.filepath.name]}")
                      prd_face_bgr = cv2.imread(precomputed_face) #take care if BW, color, ...
                      #RESIZE TO THE PROPER RESOLUTION!
                      prd_face_bgr = cv2.resize(prd_face_bgr, (192, 192), interpolation=cv2.INTER_CUBIC) #set to the proper resolution etc. 
                      cv2.imshow("PRECOMPUTED FACE", prd_face_bgr)
                      prd_face_bgr = prd_face_bgr/ 255.0 #?
                      cv2.waitKey(1)                                      
                #if not use_bw_input or colorize_pix2pix: #or colorize... #10-8-2022
                if not use_bw_input or colorize_pix2pix or use_precomputed_faces: #or use_precomputed_faces - #20-8-2022 or colorize... #10-8-2022
                    print("if not use_bw_input or colorize_pix2pix: #or colorize... #10-8-2022") 
                    out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC ) #like img_bgr? grayscale?
                else:  #The same now
                    print("else if use_bw_input or colorize_pix2pix:")
                    if debug_merge_masked: print("else: cv2.warpAffine... cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4")
                    #shape is 1,192,192 -- shoud be 192,192,1? or just 192,192?
                    #prd_face_bgr = prd_face_bgr[1:] #,np.newaxis] no    
                    print(f"prd_face_bgr.shape = {prd_face_bgr.shape}")
                    prd_face_bgr = np.squeeze(prd_face_bgr)
                    if debug_merge_masked: print(prd_face_bgr.shape)
                    
                    if has_color_frames_path:
                      out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr_color), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)#img_bgr_color -- #13-8-2022                    
                    else: out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4) #18-5-2022                        
                    
                    
                out_img = np.clip(out_img, 0.0, 1.0)

                if 'seamless' in cfg.mode:
                    try:
                        #calc same bounding rect and center point as in cv2.seamlessClone to prevent jittering (not flickering)
                        l,t,w,h = cv2.boundingRect( (img_face_seamless_mask_a*255).astype(np.uint8) )
                        s_maskx, s_masky = int(l+w/2), int(t+h/2)
                        out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), img_bgr_uint8, (img_face_seamless_mask_a*255).astype(np.uint8), (s_maskx,s_masky) , cv2.NORMAL_CLONE )
                        out_img = out_img.astype(dtype=np.float32) / 255.0
                    except Exception as e:
                        #seamlessClone may fail in some cases
                        e_str = traceback.format_exc()

                        if 'MemoryError' in e_str:
                            raise Exception("Seamless fail: " + e_str) #reraise MemoryError in order to reprocess this data by other processes
                        else:
                            print ("Seamless fail: " + e_str)

                cfg_mp = cfg.motion_blur_power / 100.0
                
                if not has_color_frames_path:
                  out_img = img_bgr*(1-img_face_mask_a) + (out_img*img_face_mask_a)
                else:
                    print("if has_color_frames_path")
                    print("out_img = img_bgr_color*(1-img_face_mask_a) + (out_img*img_face_mask_a)")                    
                    out_img = img_bgr_color*(1-img_face_mask_a) + (out_img*img_face_mask_a)
                    print(out_img)
                    #cv2.imshow("color out_img?", out_img)    #13-8-2022
                
                print(f"np.sum(out_img)={np.sum(out_img)}")           

                if ('seamless' in cfg.mode and cfg.color_transfer_mode != 0) or \
                   cfg.mode == 'seamless-hist-match' or \
                   cfg_mp != 0 or \
                   cfg.blursharpen_amount != 0 or \
                   cfg.image_denoise_power != 0 or \
                   cfg.bicubic_degrade_power != 0:

                    out_face_bgr = cv2.warpAffine( out_img, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )

                    if 'seamless' in cfg.mode and cfg.color_transfer_mode != 0:
                        if cfg.color_transfer_mode == 1:
                            out_face_bgr = imagelib.reinhard_color_transfer (out_face_bgr, dst_face_bgr, target_mask=wrk_face_mask_area_a, source_mask=wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 2: #lct
                            out_face_bgr = imagelib.linear_color_transfer (out_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 3: #mkl
                            out_face_bgr = imagelib.color_transfer_mkl (out_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 4: #mkl-m
                            out_face_bgr = imagelib.color_transfer_mkl (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 5: #idt
                            out_face_bgr = imagelib.color_transfer_idt (out_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 6: #idt-m
                            out_face_bgr = imagelib.color_transfer_idt (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 7: #sot-m
                            out_face_bgr = imagelib.color_transfer_sot (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a, steps=10, batch_size=30)
                            out_face_bgr = np.clip (out_face_bgr, 0.0, 1.0)
                        elif cfg.color_transfer_mode == 8: #mix-m
                            out_face_bgr = imagelib.color_transfer_mix (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)

                    if cfg.mode == 'seamless-hist-match':
                        out_face_bgr = imagelib.color_hist_match(out_face_bgr, dst_face_bgr, cfg.hist_match_threshold)

                    if cfg_mp != 0:
                        k_size = int(frame_info.motion_power*cfg_mp)
                        if k_size >= 1:
                            k_size = np.clip (k_size+1, 2, 50)
                            if cfg.super_resolution_power != 0:
                                k_size *= 2
                            out_face_bgr = imagelib.LinearMotionBlur (out_face_bgr, k_size , frame_info.motion_deg)

                    if cfg.blursharpen_amount != 0:
                        print("if cfg.blursharpen_amount != 0:")
                        out_face_bgr = imagelib.blursharpen ( out_face_bgr, cfg.sharpen_mode, 3, cfg.blursharpen_amount)

                    if cfg.image_denoise_power != 0:
                        n = cfg.image_denoise_power
                        while n > 0:
                            img_bgr_denoised = cv2.medianBlur(img_bgr, 5)
                            if int(n / 100) != 0:
                                img_bgr = img_bgr_denoised
                            else:
                                pass_power = (n % 100) / 100.0
                                img_bgr = img_bgr*(1.0-pass_power)+img_bgr_denoised*pass_power
                            n = max(n-10,0)

                    if cfg.bicubic_degrade_power != 0:
                        p = 1.0 - cfg.bicubic_degrade_power / 101.0
                        img_bgr_downscaled = cv2.resize (img_bgr, ( int(img_size[0]*p), int(img_size[1]*p ) ), interpolation=cv2.INTER_CUBIC)
                        img_bgr = cv2.resize (img_bgr_downscaled, img_size, interpolation=cv2.INTER_CUBIC)

                    new_out = cv2.warpAffine( out_face_bgr, face_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )
                    print(f"seamless: np.sum(new_out)={np.sum(new_out)}")           
                    out_img =  np.clip( img_bgr*(1-img_face_mask_a) + (new_out*img_face_mask_a) , 0, 1.0 )
                    print(f"seamless: np.sum(out_img)={np.sum(out_img)}")           

                if cfg.color_degrade_power != 0:
                    if use_bw_input: #23-5-2022
                       print("if cfg.color_degrade_power != 0:, if use_bw_input... out_img={out_img.shape}")
                       out_img_reduced = imagelib.reduce_colors_bw(out_img, reduce_colors_bw_number_steps)
                    else: out_img_reduced = imagelib.reduce_colors(out_img, 256)
                    if cfg.color_degrade_power == 100:
                        out_img = out_img_reduced
                    else:
                        alpha = cfg.color_degrade_power / 100.0
                        out_img = (out_img*(1.0-alpha) + out_img_reduced*alpha)
        out_merging_mask_a = img_face_mask_a
        print(f"out_merging_mask_a: np.sum(out_merging_mask_a)={np.sum(out_merging_mask_a)}") 
    if out_img is None:
        print("out_img is None!")
        out_img = img_bgr.copy()
    
    #CALL THIS IN MergeMaskeface?
    """
    if colorize_pix2pix:
        os.system("python C:\\DFL\\DeepFaceLab_DirectX12\\_internal\\DeepFaceLab\\colorize\\pix_128.py")
        #out_img = cv2.imread("T:\\predicted.png") #_image.png")
        out_img_face = cv2.imread("T:\\predicted.png") #_image.png")
        cv2.imshow("PREDICTED IN DFL", out_img)
        cv2.waitKey(1)
    
    #pix2pix now is only 128x128, model is 192
    out_img_face = cv2.resize(out_img, (192, 192), interpolation=cv2.INTER_CUBIC)
    """
    ###
        
    if debug_merge_masked: print(f"\nMergeMaskedFace: return... out_img.shape, out_merging_mask_a.shape={out_img.shape}, {out_merging_mask_a.shape}")
    print("Before return out_img, out_merging_mask_a")
    
    return out_img, out_merging_mask_a

#per frame?
def MergeMasked (predictor_func,
                 predictor_input_shape,
                 face_enhancer_func,
                 xseg_256_extract_func,
                 cfg,
                 frame_info):
    global count_color_frame
    if debug_merge_masked: print("\nMergeMasked")    
    
    #if use_bw_input: img_bgr_uint8 = cv2_imread(frame_info.filepath, 0)
    if use_bw_input and not colorize_pix2pix: img_bgr_uint8 = cv2_imread(frame_info.filepath, 0)
    #if the color background is preserved, use the same, don't convert it to bw and read again color #13-8-2022 - future
    
    #ALSO color etc. if it's colorizing    
    #else: img_bgr_uint8 = cv2_imread(frame_info.filepath) #it should be color if colorize, else - a check and merge( ...)
    
    else: #if not has_color_frames_path:
        img_bgr_uint8 = cv2_imread(frame_info.filepath) #it should be color if colorize, else -
        
    """
    else: 
         if colorize_pix2pix and has_color_frames_path: #13-8-2022
              #img_bgr_original_uint8 = cv2_imread(frame_info.filepath) -- only dimensions are needed, check the structure!
              img_bgr_uint8 = cv2.imread(color_files_list[count_color_frame])
              print("Currentframe path: ", frame_info.filepath)
              print("Colorframe path: ", color_files_list[count_color_frame])
              #color_filepath = frame_info.filepath            
              
              ho = img_bgr_uint8.shape[0]
              wo = img_bgr_uint8.shape[1]
              
              #
              aspect_original = wo/ho #img_bgr.shape[1] / img_bgr.shape[0]      
              #img_bgr = cv2.imread(color_files_list[count_color_frame])
              #
              ### img_bgr_uint8 = cv2.imread(color_files_list[count_color_frame])
              #hc, wc, cc = img_bgr.shape
              hc = img_bgr_uint8.shape[0]
              wc = img_bgr_uint8.shape[1]
              aspect_color = wc/hc #img_bgr.shape[1] / img_bgr.shape[0]
              #currently has to resize      
              #print("ASPECTS: original, 
              
              
              #scale = wo/wc #1280/1920
              scale = 1
              #print(f"ASPECTs and scale  {aspect_original}, {aspect_color}, {scale}")
              print(f"ASPECTs and scale {aspect_color}, {scale}")
              #if aspect_color!= aspect_original:
              cv2.imshow("FRAME_COLOR_SOURCE?",img_bgr_uint8) #COLOR IS COLOR
              if scale < 1:
                #should compute once etc. powers of two/divisible ... 
                #for now fixed
                #img_bgr = cv2.resize(int(scale* ...
                print("Resizing to 1280x720")
                #img_bgr = cv2.resize(img_bgr, (1280,720), cv2.INTER_CUBIC) #cv2.INTER_LANCZOS4) #SLOW BUT HIGHEST QUALITY - maybe not needed?           
                img_bgr_uint8 = cv2.resize(img_bgr_uint8, (1280,720), cv2.INTER_CUBIC) #cv2.INTER_LANCZOS4) #SLOW BUT HIGHEST QUALITY - maybe not needed?           
                #img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
                               
                #img_bgr = img_bgr_uint8/255 #divided later
                #cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
                
              count_color_frame+=1
    """  
        
    cv2.imshow("MergeMasked img_bgr FRAME?",img_bgr_uint8) #COLOR IS COLOR
    cv2.waitKey(1)
    
    if debug_merge_masked: print(f"img_bgr_uint8.shape={img_bgr_uint8.shape}")
    if not use_bw_input or colorize_pix2pix: # or colorize_
        if debug_merge_masked: print("if not use_bw_input, normalize 3") #18-5-2022
        img_bgr_uint8 = imagelib.normalize_channels (img_bgr_uint8, 3)
    else:
      img_bgr_uint8 = imagelib.normalize_channels (img_bgr_uint8, 1)
      if debug_merge_masked: print("if use_bw_input, normalize 1 (now 3, check)") 
      
    #FACE EHNHANCER?   
        
    img_bgr = img_bgr_uint8.astype(np.float32) / 255.0
    if debug_merge_masked: print(f"img_bgr_uint8.shape={img_bgr.shape}"); print(f"{img_bgr}\n\n{img_bgr_uint8}")
    
    outs = [] #landmarks from the driver frame
    for face_num, img_landmarks in enumerate( frame_info.landmarks_list ):
        out_img, out_img_merging_mask = MergeMaskedFace (predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, cfg, frame_info, img_bgr_uint8, img_bgr, img_landmarks)
        
        #cv2.imwrite("T:\\input_image.png", out_img )
        #CALL THIS IN MergeMaskeface? #10-8-2022
        """
        os.system("python C:\\DFL\\DeepFaceLab_DirectX12\\_internal\\DeepFaceLab\\colorize\\pix_128.py")
        out_img = cv2.imread("T:\\predicted.png") #_image.png")
        cv2.imshow("PREDICTED IN DFL", out_img)
        cv2.waitKey()
        """
                
        #out_img = pix_128.colorize(out_img) #maybe some conversions        
        
        if show_out_img: cv2.imshow("OUT_IMG", out_img); cv2.waitKey(1) #if ... #13-8-2022
        
        outs += [ (out_img, out_img_merging_mask) ]
        #WILL BREAK IF USING GRAYSCALE
    
    #if colorize_pix2pix: final_img
    
    #Combining multiple face outputs
    final_img = None
    final_mask = None
    for img, merging_mask in outs:
        h,w,c = img.shape

        if final_img is None:
            final_img = img
            final_mask = merging_mask
        else:
            final_img = final_img*(1-merging_mask) + img*merging_mask
            final_mask = np.clip (final_mask + merging_mask, 0, 1 )
    print(f"final_img.shape={final_img.shape}")
    final_img = np.concatenate ( [final_img, final_mask], -1)
    if colorize_pix2pix:
       cv2.imshow("Final image", final_img) #if ... #28-8-2022 - merging, BW without colorie - error, invalid number of channels, VScn::contains(scn), scn==s2
    #cv2.waitKey(1)
    #cv2.waitKey(0)

    return (final_img*255).astype(np.uint8)
