import sys
import traceback

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.interact import interact as io
from facelib import FaceType, LandmarksProcessor

is_windows = sys.platform[0:3] == 'win'
xseg_input_size = 256

use_bw_input = "use_bw_input" in os.environ  #18-5-2022

debug_merge_masked = "debug_merge_masked" in os.environ #22-5-2022

reduce_colors_bw_number_steps = 16 #16 #23-5-2022 

#print(f"default_merge_cpu_count = {default_merge_cpu_count}")

counter = 0
apply_reshape_test = "apply_reshape_test" in os.environ #25-5-2022
#apply_reshape_test = True

not_inline = True #lct test, ...

def MergeMaskedFace (predictor_func, predictor_input_shape,
                     face_enhancer_func,
                     xseg_256_extract_func,
                     cfg, frame_info, img_bgr_uint8, img_bgr, img_face_landmarks):
    if debug_merge_masked: print("MergeMaskedFace")
    if debug_merge_masked: print(f"img_bgr.shape={img_bgr.shape}")
    img_size = img_bgr.shape[1], img_bgr.shape[0]
    img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)

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
    if use_bw_input and img_bgr.shape==2:
      print("if use_bw_input and img_bgr.shape==2:")
      img_bgr = img_bgr[...,np.newaxis]
    #if user_bw_input: 
    #    dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    #cv2.INTER_LANCZOS4
    
    dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    if debug_merge_masked: print(f"AFTER: {dst_face_bgr.shape}")
    
    dst_face_bgr      = np.clip(dst_face_bgr, 0, 1)    
    
    dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

    if debug_merge_masked: print("BEFORE: {predictor_input_bgr = cv2.resize...}")
    predictor_input_bgr      = cv2.resize (dst_face_bgr, (input_size,input_size) )
    if debug_merge_masked: print(f"AFTER: predictor_input_bgr.shape=predictor_input_bgr.shape={predictor_input_bgr.shape}, dst_face_bgr.shape={dst_face_bgr.shape}, input_size={input_size}")
    if debug_merge_masked: print("BEFORE:predicted = predictor_func (predictor_input_bgr)")

    predicted = predictor_func (predictor_input_bgr)
    prd_face_bgr          = np.clip (predicted[0], 0, 1.0)
    prd_face_mask_a_0     = np.clip (predicted[1], 0, 1.0)
    prd_face_dst_mask_a_0 = np.clip (predicted[2], 0, 1.0)

    if cfg.super_resolution_power != 0: #only if !=0
        prd_face_bgr_enhanced = face_enhancer_func(prd_face_bgr, is_tanh=True, preserve_size=False)
        mod = cfg.super_resolution_power / 100.0
        prd_face_bgr = cv2.resize(prd_face_bgr, (output_size,output_size))*(1.0-mod) + prd_face_bgr_enhanced*mod
        prd_face_bgr = np.clip(prd_face_bgr, 0, 1)

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
                        
                            """ from the function
                            try:
                              prd_face_bgr = imagelib.linear_color_transfer (prd_face_bgr, dst_face_bgr)
                            except:
                                  print("Exception in linear_color_transfer!!!"+sys.exc_info());                             
                            """
                            #It doesn't reach here. WHY?! No exception, just "disappears"before Cs = ... in linear_color_transfer #26-5-2022 
                            print(f"np.sum(prd_face_bgr)={np.sum(prd_face_bgr)}")
                            #INLINED
                            print("elif cfg.color_transfer_mode == 2: #lct")                        
                            
                            print(f"LCT cfg.color_transfer_mode == 7 prd_face_bgr.shape= {prd_face_bgr.shape},dst_face_bgr.shape={dst_face_bgr.shape}")
                            if prd_face_bgr.shape[0] ==1:
                              prd_face_bgr = prd_face_bgr.transpose(1,2,0)
                              print(f"prd_face_bgr.transpose(1,2,0), prd_face_bgr.shape= {prd_face_bgr.shape}")
                            if len(dst_face_bgr.shape) ==2:
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
                            #else: matched_img = ts    
                            else: #matched_img = ts #ERROR!
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
                if not use_bw_input:
                    out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )
                else: 
                    if debug_merge_masked: print("cv2.warpAffine... cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4")
                    #shape is 1,192,192 -- shoud be 192,192,1? or just 192,192?
                    #prd_face_bgr = prd_face_bgr[1:] #,np.newaxis] no    
                    print(f"prd_face_bgr.shape = {prd_face_bgr.shape}")
                    prd_face_bgr = np.squeeze(prd_face_bgr)
                    if debug_merge_masked: print(prd_face_bgr.shape)
                    out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4) #18-5-2022
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
                
                out_img = img_bgr*(1-img_face_mask_a) + (out_img*img_face_mask_a)
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
        
    if debug_merge_masked: print(f"\nMergeMaskedFace: return... out_img.shape, out_merging_mask_a.shape={out_img.shape}, {out_merging_mask_a.shape}")
    print("Before return out_img, out_merging_mask_a")
    return out_img, out_merging_mask_a


def MergeMasked (predictor_func,
                 predictor_input_shape,
                 face_enhancer_func,
                 xseg_256_extract_func,
                 cfg,
                 frame_info):
    if debug_merge_masked: print("\nMergeMasked")
    if use_bw_input: img_bgr_uint8 = cv2_imread(frame_info.filepath, 0)
    else: img_bgr_uint8 = cv2_imread(frame_info.filepath)
    if debug_merge_masked: print(f"img_bgr_uint8.shape={img_bgr_uint8.shape}")
    if not use_bw_input:
        if debug_merge_masked: print("if not use_bw_input, normalize 3") #18-5-2022
        img_bgr_uint8 = imagelib.normalize_channels (img_bgr_uint8, 3)
    else:
      img_bgr_uint8 = imagelib.normalize_channels (img_bgr_uint8, 1)
      if debug_merge_masked: print("if use_bw_input, normalize 1 (now 3, check)") 
      
    #FACE EHNHANCER?       
    img_bgr = img_bgr_uint8.astype(np.float32) / 255.0
    if debug_merge_masked: print(f"img_bgr_uint8.shape={img_bgr.shape}")

    outs = []
    for face_num, img_landmarks in enumerate( frame_info.landmarks_list ):
        out_img, out_img_merging_mask = MergeMaskedFace (predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, cfg, frame_info, img_bgr_uint8, img_bgr, img_landmarks)
        outs += [ (out_img, out_img_merging_mask) ]

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

    final_img = np.concatenate ( [final_img, final_mask], -1)

    return (final_img*255).astype(np.uint8)
