import collections
import math
from enum import IntEnum

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.imagelib import sd
from facelib import FaceType, LandmarksProcessor

bDebugSampleProcessor = False #True #False #24-4-2022
import os
use_color_input_and_grayscale_model = "color_input_and_grayscale_model" in os.environ
#SHOULD SAVE ...  #25-4-2022
countSaves = 0
maxCountSaves = 1000
SavePath = "Z:\\dfl\\"
#cv2.imwrite(....)

use_bw_input = "use_bw_input" in os.environ

#print_samples_info = "print_samples_info" in os.environ #True #3-5-2022 Read on each function call - change on the fly?
print_samples_info = False
if "print_samples_info" in os.environ:
  if os.environ["print_samples_info"]=="1": print_samples_info = True
else: os.environ["print_samples_info"] = "0" #Create the key in order to use it for communication with the process of the preview

debug_color_transfer = "debug_color_transfer" in os.environ #26-5-2022

print(f"print_samples_info={print_samples_info}")

def SaveImage(out_sample, chType):
  global countSaves
  global maxCountSaves
  if countSaves >= maxCountSaves: return #or check before calling the function
  #cv2.imwrite(f"{SavePath}{str(countSaves)}-{chType}.jpg", out_sample)
  print(f"SaveImage: {chType}={np.sum(out_sample)}")
  countSaves+=1
  

class SampleProcessor(object):
    class SampleType(IntEnum):
        NONE = 0
        IMAGE = 1
        FACE_IMAGE = 2
        FACE_MASK  = 3
        LANDMARKS_ARRAY            = 4
        PITCH_YAW_ROLL             = 5
        PITCH_YAW_ROLL_SIGMOID     = 6

    class ChannelType(IntEnum):
        NONE = 0
        BGR                   = 1  #BGR
        G                     = 2  #Grayscale
        GGG                   = 3  #3xGrayscale

    class FaceMaskType(IntEnum):
        NONE          = 0
        FULL_FACE      = 1  # mask all hull as grayscale
        EYES           = 2  # mask eyes hull as grayscale
        EYES_MOUTH     = 3  # eyes and mouse

    class Options(object):
        def __init__(self, random_flip = True, rotation_range=[-10,10], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05] ):
            self.random_flip = random_flip
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range

    @staticmethod
    def process (samples, sample_process_options, output_sample_types, debug, ct_sample=None):
        SPST = SampleProcessor.SampleType 
        SPCT = SampleProcessor.ChannelType  #just that type would be 1C = G = 2? #22-4-2022
        SPFMT = SampleProcessor.FaceMaskType
        
        """
        print_samples_info = os.environ["print_samples_info"] #3-5-2022: press 'i' in the preview to toggle
        if print_samples_info=="1":
          #print(samples) #3-5-2022
          #for s in samples: print(s.filename, end="\t")
          for s in samples: io.log_info(f"SampleProcessor.process: {s.filename}") #, end="\t")
        """
        outputs = []
        for sample in samples:
            sample_rnd_seed = np.random.randint(0x80000000)
            
            sample_face_type = sample.face_type
            sample_bgr = sample.load_bgr()
            sample_landmarks = sample.landmarks
            ct_sample_bgr = None
            #if it's grayscale, there are only 2 dims, error 
            #h,w,c = sample_bgr.shape  #that could be kept, just c = 1? #22-4-2022
            #that should be global setting etc.!
            if len(sample_bgr.shape)==2:
              h,w = sample_bgr.shape; c = 1
              if bDebugSampleProcessor: print(f"SampleProcessor.process.BW, h={h},w={w},c=1")
            else: 
              h,w,c = sample_bgr.shape  #that could be kept, just c = 1? #22-4-2022
              if bDebugSampleProcessor: print(f"SampleProcessor.process.BW, h={h},w={w},c={c} = sample_bgr.shape({sample_bgr.shape}")
            
            def get_full_face_mask():   
                xseg_mask = sample.get_xseg_mask()                                     
                if xseg_mask is not None:           
                    if xseg_mask.shape[0] != h or xseg_mask.shape[1] != w:
                        xseg_mask = cv2.resize(xseg_mask, (w,h), interpolation=cv2.INTER_CUBIC)                    
                        xseg_mask = imagelib.normalize_channels(xseg_mask, 1)
                    return np.clip(xseg_mask, 0, 1)
                else:                    
                    full_face_mask = LandmarksProcessor.get_image_hull_mask (sample_bgr.shape, sample_landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
                    if bDebugSampleProcessor:
                      print(f"sample_bgr.shape={sample_bgr.shape}")
                      print(f"SampleProcessor.get_full_face_mask: sample_landmarks={sample_landmarks[0:1]}")
                      print(f"SampleProcessor.get_full_face_mask: np.sum(full_face_mask)={np.sum(full_face_mask)}")
                    return np.clip(full_face_mask, 0, 1)
                
            def get_eyes_mask():
                eyes_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample_landmarks)
                return np.clip(eyes_mask, 0, 1)
            
            def get_eyes_mouth_mask():                
                eyes_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample_landmarks)
                mouth_mask = LandmarksProcessor.get_image_mouth_mask (sample_bgr.shape, sample_landmarks)
                mask = eyes_mask + mouth_mask
                return np.clip(mask, 0, 1)
                
            is_face_sample = sample_landmarks is not None

            if debug and is_face_sample:
                LandmarksProcessor.draw_landmarks (sample_bgr, sample_landmarks, (0, 1, 0))

            outputs_sample = []
            for opts in output_sample_types:
                resolution     = opts.get('resolution', 0)
                sample_type    = opts.get('sample_type', SPST.NONE)
                channel_type   = opts.get('channel_type', SPCT.NONE)                
                nearest_resize_to = opts.get('nearest_resize_to', None)
                warp           = opts.get('warp', False)
                transform      = opts.get('transform', False)
                random_hsv_shift_amount = opts.get('random_hsv_shift_amount', 0)
                normalize_tanh = opts.get('normalize_tanh', False)
                ct_mode        = opts.get('ct_mode', None)
                data_format    = opts.get('data_format', 'NHWC')
                
                rnd_seed_shift      = opts.get('rnd_seed_shift', 0)
                warp_rnd_seed_shift = opts.get('warp_rnd_seed_shift', rnd_seed_shift)
                
                rnd_state      = np.random.RandomState (sample_rnd_seed+rnd_seed_shift)
                warp_rnd_state = np.random.RandomState (sample_rnd_seed+warp_rnd_seed_shift)
                
                warp_params = imagelib.gen_warp_params(resolution, 
                                                       sample_process_options.random_flip, 
                                                       rotation_range=sample_process_options.rotation_range, 
                                                       scale_range=sample_process_options.scale_range, 
                                                       tx_range=sample_process_options.tx_range, 
                                                       ty_range=sample_process_options.ty_range, 
                                                       rnd_state=rnd_state,
                                                       warp_rnd_state=warp_rnd_state,
                                                       )
                
                if sample_type == SPST.FACE_MASK or sample_type == SPST.IMAGE: 
                    border_replicate = False
                elif sample_type == SPST.FACE_IMAGE:
                    border_replicate = True
                    
                    
                border_replicate = opts.get('border_replicate', border_replicate)
                borderMode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
                
                
                if sample_type == SPST.FACE_IMAGE or sample_type == SPST.FACE_MASK:
                    if not is_face_sample:    
                        raise ValueError("face_samples should be provided for sample_type FACE_*")

                if sample_type == SPST.FACE_IMAGE or sample_type == SPST.FACE_MASK:
                    face_type      = opts.get('face_type', None)
                    face_mask_type = opts.get('face_mask_type', SPFMT.NONE)
                
                    if face_type is None:
                        raise ValueError("face_type must be defined for face samples")

                    if sample_type == SPST.FACE_MASK: 
                        if face_mask_type == SPFMT.FULL_FACE:
                            img = get_full_face_mask()
                        elif face_mask_type == SPFMT.EYES:
                            img = get_eyes_mask()
                        elif face_mask_type == SPFMT.EYES_MOUTH:
                            mask = get_full_face_mask().copy()
                            mask[mask != 0.0] = 1.0                            
                            img = get_eyes_mouth_mask()*mask
                        else:
                            img = np.zeros ( sample_bgr.shape[0:2]+(1,), dtype=np.float32)

                        if sample_face_type == FaceType.MARK_ONLY:
                            raise NotImplementedError()
                            mat  = LandmarksProcessor.get_transform_mat (sample_landmarks, warp_resolution, face_type)
                            img = cv2.warpAffine( img, mat, (warp_resolution, warp_resolution), flags=cv2.INTER_LINEAR )
                            
                            img = imagelib.warp_by_params (warp_params, img, warp, transform, can_flip=True, border_replicate=border_replicate, cv2_inter=cv2.INTER_LINEAR)
                            img = cv2.resize( img, (resolution,resolution), interpolation=cv2.INTER_LINEAR )
                        else:
                            if face_type != sample_face_type:
                                mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, face_type)                            
                                img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_LINEAR )
                            else:
                                if w != resolution:
                                    img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_LINEAR )
                                
                            img = imagelib.warp_by_params (warp_params, img, warp, transform, can_flip=True, border_replicate=border_replicate, cv2_inter=cv2.INTER_LINEAR)

                        if face_mask_type == SPFMT.EYES_MOUTH:
                            div = img.max()
                            if div != 0.0:
                                img = img / div # normalize to 1.0 after warp
                            
                        if len(img.shape) == 2:
                            img = img[...,None]
                        
                        #if bDebugSampleProcessor: SaveImage(out_sample,"FACE_MASK_G") #25-4-2022  #commented 27-4-2022 
                        if channel_type == SPCT.G:
                            out_sample = img.astype(np.float32) #Initially it is not float32
                        else:
                            raise ValueError("only channel_type.G supported for the mask")

                    elif sample_type == SPST.FACE_IMAGE:
                        #if bDebugSampleProcessor: print("\nelif sample_type == SPST.FACE_IMAGE:")
                        img = sample_bgr                      
                            
                        if face_type != sample_face_type:
                            mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, face_type)
                            img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_CUBIC )
                        else:
                            if w != resolution:
                                img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_CUBIC )
                        #cv2.imshow("== SPST.FACE_IMAGE", img*255)
                        # Apply random color transfer                        
                        if ct_mode is not None and ct_sample is not None:
                            if ct_sample_bgr is None:
                               ct_sample_bgr = ct_sample.load_bgr()
                               
                            if len(ct_sample_bgr.shape)==2:      
                                ct_sample_bgr = ct_sample_bgr[...,np.newaxis]
                                if debug_color_transfer: print(f"ct_sample_bgr Add axis to BW image, {ct_sample_bgr.shape}")
                                
                            #if use_bw_input:
                            if len(img.shape)==2:   
                              if debug_color_transfer:  print(f"SampleProcessor.py: img.shape = {img.shape}")
                              if len(img.shape)==2:      
                                img = img[...,np.newaxis]
                                if debug_color_transfer:  print(f"SampleProcessor img-->Add axis to BW image, {img.shape}")
                                 
                            img = imagelib.color_transfer (ct_mode, img, cv2.resize( ct_sample_bgr, (resolution,resolution), interpolation=cv2.INTER_LINEAR ) )
                        
                        if random_hsv_shift_amount != 0: #don't use for BW!
                            a = random_hsv_shift_amount
                            h_amount = max(1, int(360*a*0.5))
                            img_h, img_s, img_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                            img_h = (img_h + rnd_state.randint(-h_amount, h_amount+1) ) % 360
                            img_s = np.clip (img_s + (rnd_state.random()-0.5)*a, 0, 1 )
                            img_v = np.clip (img_v + (rnd_state.random()-0.5)*a, 0, 1 )
                            img = np.clip( cv2.cvtColor(cv2.merge([img_h, img_s, img_v]), cv2.COLOR_HSV2BGR) , 0, 1 )

                        img  = imagelib.warp_by_params (warp_params, img,  warp, transform, can_flip=True, border_replicate=border_replicate)
  
                        img = np.clip(img.astype(np.float32), 0, 1)

                        # Transform from BGR to desired channel_type
                        if channel_type == SPCT.BGR:
                            out_sample = img
                            if bDebugSampleProcessor: SaveImage(out_sample,"BGR")
                        elif channel_type == SPCT.G:
                            #if len(img.shape) == 3: #22-4-2022 96,96,1
                            """ SO FAR, #25-4-2022, 8:33 --> try with the original
                            if img.shape[2] == 3: #22-4-2022 96,96,1  else just don't convert
                              print("IMAGE.SHAPE?=", img.shape)
                              out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                            else:                              
                            out_sample = img; print("else: out_sample = img; channel_type==SPCT.G");
                            """
                            #out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                            
                            """
                            elif sample_type == SPST.FACE_IMAGE:Traceback (most recent call last):
                            File "C:\DFL\DeepFaceLab_DirectX12\_internal\DeepFaceLab\samplelib\SampleGeneratorFace.py", line 134, in batch_func
                               x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug, ct_sample=ct_sample)
                               File "C:\DFL\DeepFaceLab_DirectX12\_internal\DeepFaceLab\samplelib\SampleProcessor.py", line 231, in process
                            out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                            cv2.error: OpenCV(4.1.0) c:\projects\opencv-python\opencv\modules\imgproc\src\color.simd_helpers.hpp:92: error: (-2:Unspecified error) in function '__cdecl cv::impl::`anonymous-namespace'::CvtHelper<struct cv::impl::`anonymous namespace'::Set<3,4,-1>,struct cv::impl::A0xe227985e::Set<1,-1,-1>,struct cv::impl::A0xe227985e::Set<0,2,5>,2>::CvtHelper(const class cv::_InputArray &,const class cv::_OutputArray &,int)'
                            > Invalid number of channels in input image:
                            >     'VScn::contains(scn)'
                            > where
                            >     'scn' is 1

                            """
                            #out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                            out_sample = img
                            if bDebugSampleProcessor:
                               print(f"SampleProcesor: elif channel_type == SPCT.G, np.sum(out_sample)={np.sum(out_sample)}")
                            if bDebugSampleProcessor: SaveImage(out_sample,"G")
                            
                            
                        elif channel_type == SPCT.GGG:
                            out_sample = np.repeat ( np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),-1), (3,), -1)
                            if bDebugSampleProcessor: SaveImage(out_sample, "GGG")

                    # Final transformations
                    if nearest_resize_to is not None:
                        out_sample = cv2_resize(out_sample, (nearest_resize_to,nearest_resize_to), interpolation=cv2.INTER_NEAREST)
                        if bDebugSampleProcessor: SaveImage(out_sample, "nearest_resize", bDontIncrement)
                        
                    if not debug:
                        if normalize_tanh:
                            out_sample = np.clip (out_sample * 2.0 - 1.0, -1.0, 1.0)
                    if data_format == "NCHW":
                        out_sample = np.transpose(out_sample, (2,0,1) )
                elif sample_type == SPST.IMAGE:
                    img = sample_bgr      
                    img  = imagelib.warp_by_params (warp_params, img,  warp, transform, can_flip=True, border_replicate=True)
                    img  = cv2.resize( img,  (resolution, resolution), interpolation=cv2.INTER_CUBIC )
                    out_sample = img
                    if bDebugSampleProcessor: cv2.imshow("elif sample_type == SPST.IMAGE", img) #24-4-2022 
                    #SHOULD SAVE! count ... first N files etc.! global #25-4-2022
                    
                    
                    if data_format == "NCHW":
                        out_sample = np.transpose(out_sample, (2,0,1) )
                    
                    
                elif sample_type == SPST.LANDMARKS_ARRAY:
                    l = sample_landmarks
                    l = np.concatenate ( [ np.expand_dims(l[:,0] / w,-1), np.expand_dims(l[:,1] / h,-1) ], -1 )
                    l = np.clip(l, 0.0, 1.0)
                    out_sample = l
                elif sample_type == SPST.PITCH_YAW_ROLL or sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                    pitch,yaw,roll = sample.get_pitch_yaw_roll()
                    if warp_params['flip']:
                        yaw = -yaw

                    if sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                        pitch = np.clip( (pitch / math.pi) / 2.0 + 0.5, 0, 1)
                        yaw   = np.clip( (yaw / math.pi) / 2.0 + 0.5, 0, 1)
                        roll  = np.clip( (roll / math.pi) / 2.0 + 0.5, 0, 1)

                    out_sample = (pitch, yaw)
                else:
                    raise ValueError ('expected sample_type')

                outputs_sample.append ( out_sample )
            outputs += [outputs_sample]

        return outputs

