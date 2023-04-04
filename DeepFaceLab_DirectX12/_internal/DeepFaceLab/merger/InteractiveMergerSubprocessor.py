import multiprocessing
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from merger import MergeFaceAvatar, MergeMasked, MergerConfig

from .MergerScreen import Screen, ScreenManager

# "ARNOLDIFIER" (Deepfacelab-SAEHDBW) by Twenkid (Todor Arnaudov) based on the original code by iperov et al.: DeepfaceLab
# Experimental version for colorizing the grayscale merged faces - put this file in the respective folder.
# Note that Arnoldifier currently doesn't support Interactive merging!
# Only automatic merging is supported!
# See also _internal\DeepFaceLab\merger\MergeMasked.py
# See also instructions of the required bat for the colorizing workflow and training the colorizing model
# See also pix2pix ... 

# Todor/Twenkid, 19-4-2022
merge_file_extension = ".jpg" #default was .png hardcoded - but like that there is no
#no adj. for the quality  -- it is 95%
merge_mask_extension = ".png"
merge_file_jpg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 90] #90] #90, 95, 100: smaller for tests
MERGER_DEBUG = False # True #False

use_bw_input = "use_bw_input" in os.environ #30-4-2022, Twenkid. Test!
print(f"InteractiveMergerSubprocessor.use_bw_input={use_bw_input}")

colorize_pix2pix = "colorize_pix2pix" in os.environ #9-8-2022 - 10-8-2022


import os

if "merge_file_extension" in os.environ:  #30-4-2022
   merge_file_extension = os.environ[merge_file_extension]
else: merge_file_extension = ".jpg"

if "merge_file_jpg_quality" in os.environ:
   merge_file_jpg_quality = int(os.environ["merge_file_jpg_quality"])
else: merge_file_jpg_quality = 90

if "merger_debug" in os.environ: MERGER_DEBUG = int(os.environ["merger_debug"]==1)

print(f"InteractiveMergerSubprocessor.py: merge_file_jpg_quality = merge_file_jpg_quality in os.environ {merge_file_jpg_quality}")
print(f"merge_file_jpg_quality = os.environ[merge_file_jpg_quality]: {merge_file_jpg_quality}")
print(f"MERGER_DEBUG = os.environ[merger_debug], MERGER_DEBUG = {MERGER_DEBUG}")

merge_to_color = True #18-5-2022


class InteractiveMergerSubprocessor(Subprocessor):

    class Frame(object):
        def __init__(self, prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None):
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = None
            self.output_mask_filepath = None

            self.idx = None
            self.cfg = None
            self.is_done = False
            self.is_processing = False
            self.is_shown = False
            self.image = None

    class ProcessingFrame(object):
        def __init__(self, idx=None,
                           cfg=None,
                           prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None,
                           output_filepath=None,
                           output_mask_filepath=None,
                           need_return_image = False):
            self.idx = idx
            self.cfg = cfg
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = output_filepath
            self.output_mask_filepath = output_mask_filepath

            self.need_return_image = need_return_image
            if self.need_return_image:
                self.image = None

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.log_info ('Running on %s.' % (client_dict['device_name']) )
            self.device_idx  = client_dict['device_idx']
            self.device_name = client_dict['device_name']
            self.predictor_func = client_dict['predictor_func']
            self.predictor_input_shape = client_dict['predictor_input_shape']
            self.face_enhancer_func = client_dict['face_enhancer_func']
            self.xseg_256_extract_func = client_dict['xseg_256_extract_func']


            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None:
                sys.stdin = os.fdopen(stdin_fd)

            return None

        #override
        def process_data(self, pf): #pf=ProcessingFrame
            print("InteractiveMerger... process_data")
            cfg = pf.cfg.copy()
            
            frame_info = pf.frame_info
            filepath = frame_info.filepath #set to merge_file_extension #Twenkid 19.4.2022
            print(f"filepath={filepath}")

            if len(frame_info.landmarks_list) == 0:
                
                if cfg.mode == 'raw-predict':  
                    print(f"if cfg.mode == 'raw-predict':")
                    if use_bw_input: #18-5-2022
                      h,w,c = self.predictor_input_shape
                      if merge_to_color: img_bgr = np.zeros( (h,w,3), dtype=np.uint8)
                      else: img_bgr = np.zeros( (h,w,1), dtype=np.uint8)                      
                      img_mask = np.zeros( (h,w,1), dtype=np.uint8)
                    else:
                         h,w,c = self.predictor_input_shape
                         img_bgr = np.zeros( (h,w,3), dtype=np.uint8)
                         img_mask = np.zeros( (h,w,1), dtype=np.uint8)               
                else:                
                    self.log_info (f'no faces found for {filepath.name}, copying without faces')
                    #if extract_to_bw: #28-5-2022 from color input
                    if extract_to_bw and not colorize_pix2pix:  #28-5-2022 from color input; ant not colorize_... #10-8-2022
                      img_bgr = cv2_imread_color_as_grayscale(filepath)
                      if len(img_bgr.shape)==2: img_bgr = img_bgr[..., np.newaxis] 
                    else:
                        img_bgr = cv2_imread(filepath)  #color input
                    print(f"InteractiveMergerSubprocessor.py, img_bgr = cv2_imread(filepath)=\n{img_bgr.shape}")
                    if use_bw_input and not colorize_pix2pix: #and not colorize_pix2pix - #10-8-2022
                        #if not merge_to_color:
                        #  imagelib.normalize_channels(img_bgr, 1)
                        #else:
                        #        print("Merge to color with bw input!")
                        #        imagelib.normalize_channels(img_bgr, 3)
                        if len(img_bgr.shape)==2:
                          h,w = img_bgr.shape
                          img_bgr = img_bgr[:,:,np.newaxis]
                          c = 1                        
                        else:
                            h,w,c = img_bgr.shape
                        print(f"use_bw_input, imagelib.normalize_channels(img_bgr, 1), h,w,c = {h},{w}, {c}")
                        imagelib.normalize_channels(img_bgr, 1)
                    else:
                      imagelib.normalize_channels(img_bgr, 3)
                      h,w,c = img_bgr.shape
                      
                    img_mask = np.zeros( (h,w,1), dtype=img_bgr.dtype)
                """
                #cv2_imwrite (pf.output_filepath, img_bgr)                
                try:
                    if merge_file_extension == ".jpg": #Twenkid, 19.4.2022
                        cv2_imwrite (pf.output_filepath, img_bgr, merge_file_jpg_quality)
                        print("cv2_imwrite (pf.output_filepath, img_bgr, merge_file_jpg_quality)")
                    else:
                         print("else...? cv2_imwrite (pf.output_filepath, img_bgr)")
                         cv2_imwrite (pf.output_filepath, img_bgr, merge_file_jpg_quality)
                         #cv2_imwrite (pf.output_filepath, img_bgr)                     
                except: print("Exception with cv2_imwrite..."); print(sys.exc_info());
                """
                print(f"SAVE MASK:{pf.output_mask_filepath}")
                cv2_imwrite (pf.output_mask_filepath, img_mask)

                if pf.need_return_image:
                    pf.image = np.concatenate ([img_bgr, img_mask], axis=-1)

            else:
                if cfg.type == MergerConfig.TYPE_MASKED:
                    try:
                        final_img = MergeMasked (self.predictor_func, self.predictor_input_shape,
                                                 face_enhancer_func=self.face_enhancer_func,
                                                 xseg_256_extract_func=self.xseg_256_extract_func,
                                                 cfg=cfg,
                                                 frame_info=frame_info)
                    except Exception as e:
                        e_str = traceback.format_exc()
                        if 'MemoryError' in e_str:
                            raise Subprocessor.SilenceException
                        else:
                            raise Exception( f'Error while merging file [{filepath}]: {e_str}' )

                elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
                    final_img = MergeFaceAvatar (self.predictor_func, self.predictor_input_shape,
                                                   cfg, pf.prev_temporal_frame_infos,
                                                        pf.frame_info,
                                                        pf.next_temporal_frame_infos )
                    print(f"InteractiveMergerSubprocessor.py: cv2_imwrite(...) final_img.shape={final_img.shape}")

                #cv2_imwrite (pf.output_filepath,      final_img[...,0:3] )
                #cv2_imwrite (pf.output_mask_filepath, final_img[...,3:4] )
                print(f"InteractiveMergerSubprocessor.py: cv2_imwrite(...) final_img.shape={final_img.shape}")
                if use_bw_input and not merge_to_color: #18-5-2022
                    print("cv2_imwrite...")
                    cv2.imshow("MERGE", final_img[0])
                    cv2.imshow("MASK", final_img[1])
                    cv2.waitKey(1)
                    cv2_imwrite (pf.output_filepath, final_img[0], merge_file_jpg_quality)
                    cv2_imwrite (pf.output_mask_filepath, final_img[1], merge_file_jpg_quality)
                else:                    
                    #if len(final_img.shape)==2:
                    if final_img.shape[2]==2: #should be 4 for color
                        print(f"np.sum(final_img[...,0:1], [1:2], { np.sum(final_img[:,:,0])}, {np.sum(final_img[:,:,1])}")
                        
                        mer = cv2.merge( (final_img[:, :, 0:1], final_img[:, :, 0:1],final_img[:, :, 0:1]))
                        cv2_imwrite (pf.output_filepath, mer, merge_file_jpg_quality)
                        
                        #cv2_imwrite (pf.output_filepath, final_img[:, :, 0:1], merge_file_jpg_quality)
                        cv2_imwrite (pf.output_mask_filepath, final_img[:,:,1:2], merge_file_jpg_quality)
                    else:
                        cv2_imwrite (pf.output_filepath,      final_img[...,0:3], merge_file_jpg_quality)
                        cv2_imwrite (pf.output_mask_filepath, final_img[...,3:4], merge_file_jpg_quality)

                if pf.need_return_image:
                    pf.image = final_img

            return pf

        #overridable
        def get_data_name (self, pf):
            #return string identificator of your data
            return pf.frame_info.filepath




    #override
    def __init__(self, is_interactive, merger_session_filepath, predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, merger_config, frames, frames_root_path, output_path, output_mask_path, model_iter, subprocess_count=4):
        if len (frames) == 0:
            raise ValueError ("len (frames) == 0")

        super().__init__('Merger', InteractiveMergerSubprocessor.Cli, io_loop_sleep_time=0.001)

        self.is_interactive = is_interactive
        self.merger_session_filepath = Path(merger_session_filepath)
        self.merger_config = merger_config

        self.predictor_func = predictor_func
        self.predictor_input_shape = predictor_input_shape

        self.face_enhancer_func = face_enhancer_func
        self.xseg_256_extract_func = xseg_256_extract_func

        self.frames_root_path = frames_root_path
        self.output_path = output_path
        self.output_mask_path = output_mask_path
        self.model_iter = model_iter

        self.prefetch_frame_count = self.process_count = subprocess_count

        session_data = None
        if self.is_interactive and self.merger_session_filepath.exists():
            io.input_skip_pending()
            if io.input_bool ("Use saved session?", True):
                try:
                    with open( str(self.merger_session_filepath), "rb") as f:
                        session_data = pickle.loads(f.read())

                except Exception as e:
                    pass

        rewind_to_frame_idx = None
        self.frames = frames
        self.frames_idxs = [ *range(len(self.frames)) ]
        self.frames_done_idxs = []

        if self.is_interactive and session_data is not None:
            # Loaded session data, check it
            s_frames = session_data.get('frames', None)
            s_frames_idxs = session_data.get('frames_idxs', None)
            s_frames_done_idxs = session_data.get('frames_done_idxs', None)
            s_model_iter = session_data.get('model_iter', None)

            frames_equal = (s_frames is not None) and \
                           (s_frames_idxs is not None) and \
                           (s_frames_done_idxs is not None) and \
                           (s_model_iter is not None) and \
                           (len(frames) == len(s_frames)) # frames count must match

            if frames_equal:
                for i in range(len(frames)):
                    frame = frames[i]
                    s_frame = s_frames[i]
                    # frames filenames must match
                    if frame.frame_info.filepath.name != s_frame.frame_info.filepath.name:
                        frames_equal = False
                    if not frames_equal:
                        break

            if frames_equal:
                io.log_info ('Using saved session from ' + '/'.join (self.merger_session_filepath.parts[-2:]) )

                for frame in s_frames:
                    if frame.cfg is not None:
                        # recreate MergerConfig class using constructor with get_config() as dict params
                        # so if any new param will be added, old merger session will work properly
                        frame.cfg = frame.cfg.__class__( **frame.cfg.get_config() )

                self.frames = s_frames
                self.frames_idxs = s_frames_idxs
                self.frames_done_idxs = s_frames_done_idxs

                if self.model_iter != s_model_iter:
                    # model was more trained, recompute all frames
                    rewind_to_frame_idx = -1
                    for frame in self.frames:
                        frame.is_done = False
                elif len(self.frames_idxs) == 0:
                    # all frames are done?
                    rewind_to_frame_idx = -1

                if len(self.frames_idxs) != 0:
                    cur_frame = self.frames[self.frames_idxs[0]]
                    cur_frame.is_shown = False

            if not frames_equal:
                session_data = None

        if session_data is None:
            for filename in pathex.get_image_paths(self.output_path): #remove all images in output_path
                Path(filename).unlink()

            for filename in pathex.get_image_paths(self.output_mask_path): #remove all images in output_mask_path
                Path(filename).unlink()


            frames[0].cfg = self.merger_config.copy()

        for i in range( len(self.frames) ):
            frame = self.frames[i]
            frame.idx = i
            frame.output_filepath      = self.output_path      / ( frame.frame_info.filepath.stem + merge_file_extension) #'.png' )
            frame.output_mask_filepath = self.output_mask_path / ( frame.frame_info.filepath.stem + merge_mask_extension) # '.png' ) #mask should/could be .png, small files anyway

            if not frame.output_filepath.exists() or \
               not frame.output_mask_filepath.exists():
                # if some frame does not exist, recompute and rewind
                frame.is_done = False
                frame.is_shown = False

                if rewind_to_frame_idx is None:
                    rewind_to_frame_idx = i-1
                else:
                    rewind_to_frame_idx = min(rewind_to_frame_idx, i-1)

        if rewind_to_frame_idx is not None:
            while len(self.frames_done_idxs) > 0:
                if self.frames_done_idxs[-1] > rewind_to_frame_idx:
                    prev_frame = self.frames[self.frames_done_idxs.pop()]
                    self.frames_idxs.insert(0, prev_frame.idx)
                else:
                    break
    #override
    def process_info_generator(self):
        r = [0] if MERGER_DEBUG else range(self.process_count)

        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'predictor_func': self.predictor_func,
                                      'predictor_input_shape' : self.predictor_input_shape,
                                      'face_enhancer_func': self.face_enhancer_func,
                                      'xseg_256_extract_func' : self.xseg_256_extract_func,
                                      'stdin_fd': sys.stdin.fileno() if MERGER_DEBUG else None
                                      }

    #overridable optional
    def on_clients_initialized(self):
        io.progress_bar ("Merging", len(self.frames_idxs)+len(self.frames_done_idxs), initial=len(self.frames_done_idxs) )

        self.process_remain_frames = not self.is_interactive
        self.is_interactive_quitting = not self.is_interactive

        if self.is_interactive:
            help_images = {
                    MergerConfig.TYPE_MASKED :      cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_masked.jpg') ),
                    MergerConfig.TYPE_FACE_AVATAR : cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_face_avatar.jpg') ),
                }

            self.main_screen = Screen(initial_scale_to_width=1368, image=None, waiting_icon=True)
            self.help_screen = Screen(initial_scale_to_height=768, image=help_images[self.merger_config.type], waiting_icon=False)
            self.screen_manager = ScreenManager( "Merger", [self.main_screen, self.help_screen], capture_keys=True )
            self.screen_manager.set_current (self.help_screen)
            self.screen_manager.show_current()

            self.masked_keys_funcs = {
                    '`' : lambda cfg,shift_pressed: cfg.set_mode(0),
                    '1' : lambda cfg,shift_pressed: cfg.set_mode(1),
                    '2' : lambda cfg,shift_pressed: cfg.set_mode(2),
                    '3' : lambda cfg,shift_pressed: cfg.set_mode(3),
                    '4' : lambda cfg,shift_pressed: cfg.set_mode(4),
                    '5' : lambda cfg,shift_pressed: cfg.set_mode(5),
                    '6' : lambda cfg,shift_pressed: cfg.set_mode(6),
                    'q' : lambda cfg,shift_pressed: cfg.add_hist_match_threshold(1 if not shift_pressed else 5),
                    'a' : lambda cfg,shift_pressed: cfg.add_hist_match_threshold(-1 if not shift_pressed else -5),
                    'w' : lambda cfg,shift_pressed: cfg.add_erode_mask_modifier(1 if not shift_pressed else 5),
                    's' : lambda cfg,shift_pressed: cfg.add_erode_mask_modifier(-1 if not shift_pressed else -5),
                    'e' : lambda cfg,shift_pressed: cfg.add_blur_mask_modifier(1 if not shift_pressed else 5),
                    'd' : lambda cfg,shift_pressed: cfg.add_blur_mask_modifier(-1 if not shift_pressed else -5),
                    'r' : lambda cfg,shift_pressed: cfg.add_motion_blur_power(1 if not shift_pressed else 5),
                    'f' : lambda cfg,shift_pressed: cfg.add_motion_blur_power(-1 if not shift_pressed else -5),
                    't' : lambda cfg,shift_pressed: cfg.add_super_resolution_power(1 if not shift_pressed else 5),
                    'g' : lambda cfg,shift_pressed: cfg.add_super_resolution_power(-1 if not shift_pressed else -5),
                    'y' : lambda cfg,shift_pressed: cfg.add_blursharpen_amount(1 if not shift_pressed else 5),
                    'h' : lambda cfg,shift_pressed: cfg.add_blursharpen_amount(-1 if not shift_pressed else -5),
                    'u' : lambda cfg,shift_pressed: cfg.add_output_face_scale(1 if not shift_pressed else 5),
                    'j' : lambda cfg,shift_pressed: cfg.add_output_face_scale(-1 if not shift_pressed else -5),
                    'i' : lambda cfg,shift_pressed: cfg.add_image_denoise_power(1 if not shift_pressed else 5),
                    'k' : lambda cfg,shift_pressed: cfg.add_image_denoise_power(-1 if not shift_pressed else -5),
                    'o' : lambda cfg,shift_pressed: cfg.add_bicubic_degrade_power(1 if not shift_pressed else 5),
                    'l' : lambda cfg,shift_pressed: cfg.add_bicubic_degrade_power(-1 if not shift_pressed else -5),
                    'p' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(1 if not shift_pressed else 5),
                    ';' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(-1),
                    ':' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(-5),
                    'z' : lambda cfg,shift_pressed: cfg.toggle_masked_hist_match(),
                    'x' : lambda cfg,shift_pressed: cfg.toggle_mask_mode(),
                    'c' : lambda cfg,shift_pressed: cfg.toggle_color_transfer_mode(),
                    'n' : lambda cfg,shift_pressed: cfg.toggle_sharpen_mode(),
                    }
            self.masked_keys = list(self.masked_keys_funcs.keys())

    #overridable optional
    def on_clients_finalized(self):
        io.progress_bar_close()

        if self.is_interactive:
            self.screen_manager.finalize()

            for frame in self.frames:
                frame.output_filepath = None
                frame.output_mask_filepath = None
                frame.image = None

            session_data = {
                'frames': self.frames,
                'frames_idxs': self.frames_idxs,
                'frames_done_idxs': self.frames_done_idxs,
                'model_iter' : self.model_iter,
            }
            self.merger_session_filepath.write_bytes( pickle.dumps(session_data) )

            io.log_info ("Session is saved to " + '/'.join (self.merger_session_filepath.parts[-2:]) )

    #override
    def on_tick(self):
        if MERGER_DEBUG: print("\nInteractiveMerger... on_tick")
        io.process_messages()

        go_prev_frame = False
        go_first_frame = False
        go_prev_frame_overriding_cfg = False
        go_first_frame_overriding_cfg = False

        go_next_frame = self.process_remain_frames
        go_next_frame_overriding_cfg = False
        go_last_frame_overriding_cfg = False

        cur_frame = None
        if len(self.frames_idxs) != 0:
            cur_frame = self.frames[self.frames_idxs[0]]

        if self.is_interactive:
            if MERGER_DEBUG: print("if self.is_interactive:")
            screen_image = None if self.process_remain_frames else \
                                   self.main_screen.get_image()

            self.main_screen.set_waiting_icon( self.process_remain_frames or \
                                               self.is_interactive_quitting )

            if cur_frame is not None and not self.is_interactive_quitting:

                if not self.process_remain_frames:
                    if cur_frame.is_done:
                        if not cur_frame.is_shown:
                            if cur_frame.image is None:
                                print(f"if cur_frame.image is None:, output_filepath={cur_frame.output_filepath}")
                                image      = cv2_imread (cur_frame.output_filepath, verbose=False)
                                image_mask = cv2_imread (cur_frame.output_mask_filepath, verbose=False)
                                if image is None or image_mask is None:
                                    # unable to read? recompute then
                                    #cur_frame.is_done = False
                                    #commented #26-5-2022
                                    pass
                                else:
                                    if use_bw_input: 
                                      print("InteractiveMergerSubprocessor.py: Normalize channels(image_mask,1)")
                                      image_mask = imagelib.normalize_channels(image_mask, 1) #30-4-2022
                                    else:
                                        image = imagelib.normalize_channels(image, 3)
                                    image_mask = imagelib.normalize_channels(image_mask, 1)
                                    cur_frame.image = np.concatenate([image, image_mask], -1)

                            if cur_frame.is_done:
                                io.log_info (cur_frame.cfg.to_string( cur_frame.frame_info.filepath.name) )
                                cur_frame.is_shown = True
                                screen_image = cur_frame.image
                    else:
                        self.main_screen.set_waiting_icon(True)

            self.main_screen.set_image(screen_image)
            self.screen_manager.show_current()

            key_events = self.screen_manager.get_key_events()
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == 9: #tab
                self.screen_manager.switch_screens()
            else:
                if key == 27: #esc
                    self.is_interactive_quitting = True
                elif self.screen_manager.get_current() is self.main_screen:

                    if self.merger_config.type == MergerConfig.TYPE_MASKED and chr_key in self.masked_keys:
                        self.process_remain_frames = False

                        if cur_frame is not None:
                            cfg = cur_frame.cfg
                            prev_cfg = cfg.copy()

                            if cfg.type == MergerConfig.TYPE_MASKED:
                                self.masked_keys_funcs[chr_key](cfg, shift_pressed)

                            if prev_cfg != cfg:
                                io.log_info ( cfg.to_string(cur_frame.frame_info.filepath.name) )
                                cur_frame.is_done = False
                                cur_frame.is_shown = False
                    else:

                        if chr_key == ',' or chr_key == 'm':
                            self.process_remain_frames = False
                            go_prev_frame = True

                            if chr_key == ',':
                                if shift_pressed:
                                    go_first_frame = True

                            elif chr_key == 'm':
                                if not shift_pressed:
                                    go_prev_frame_overriding_cfg = True
                                else:
                                    go_first_frame_overriding_cfg = True

                        elif chr_key == '.' or chr_key == '/':
                            self.process_remain_frames = False
                            go_next_frame = True

                            if chr_key == '.':
                                if shift_pressed:
                                    self.process_remain_frames = not self.process_remain_frames

                            elif chr_key == '/':
                                if not shift_pressed:
                                    go_next_frame_overriding_cfg = True
                                else:
                                    go_last_frame_overriding_cfg = True

                        elif chr_key == '-':
                            self.screen_manager.get_current().diff_scale(-0.1)
                        elif chr_key == '=':
                            self.screen_manager.get_current().diff_scale(0.1)
                        elif chr_key == 'v':
                            self.screen_manager.get_current().toggle_show_checker_board()

        if go_prev_frame: 
            print("if go_prev_frame: ")
            if cur_frame is None or cur_frame.is_done:
                if cur_frame is not None:
                    cur_frame.image = None

                while True:
                    if len(self.frames_done_idxs) > 0:
                        prev_frame = self.frames[self.frames_done_idxs.pop()]
                        self.frames_idxs.insert(0, prev_frame.idx)
                        prev_frame.is_shown = False
                        io.progress_bar_inc(-1)

                        if cur_frame is not None and (go_prev_frame_overriding_cfg or go_first_frame_overriding_cfg):
                            if prev_frame.cfg != cur_frame.cfg:
                                prev_frame.cfg = cur_frame.cfg.copy()
                                prev_frame.is_done = False
                        print(f"cur_frame = prev_frame, {cur_frame.shape},{prev_frame.shape} ")
                        cur_frame = prev_frame

                    if go_first_frame_overriding_cfg or go_first_frame:
                        if len(self.frames_done_idxs) > 0:
                            continue
                    break

        elif go_next_frame:
            if MERGER_DEBUG: print(f"elif go_next_frame: cur_frame.is_done={cur_frame.is_done}")
            if cur_frame is not None and cur_frame.is_done:
                cur_frame.image = None
                cur_frame.is_shown = True
                self.frames_done_idxs.append(cur_frame.idx)
                self.frames_idxs.pop(0)
                io.progress_bar_inc(1)

                f = self.frames

                if len(self.frames_idxs) != 0:
                    next_frame = f[ self.frames_idxs[0] ]
                    next_frame.is_shown = False

                    if go_next_frame_overriding_cfg or go_last_frame_overriding_cfg:

                        if go_next_frame_overriding_cfg:
                            to_frames = next_frame.idx+1
                        else:
                            to_frames = len(f)

                        for i in range( next_frame.idx, to_frames ):
                            f[i].cfg = None

                    for i in range( min(len(self.frames_idxs), self.prefetch_frame_count) ):
                        frame = f[ self.frames_idxs[i] ]
                        if frame.cfg is None:
                            if i == 0:
                                frame.cfg = cur_frame.cfg.copy()
                            else:
                                frame.cfg = f[ self.frames_idxs[i-1] ].cfg.copy()

                            frame.is_done = False #initiate solve again
                            frame.is_shown = False

            if len(self.frames_idxs) == 0:
                self.process_remain_frames = False

        return (self.is_interactive and self.is_interactive_quitting) or \
               (not self.is_interactive and self.process_remain_frames == False)


    #override
    def on_data_return (self, host_dict, pf):
        frame = self.frames[pf.idx]
        frame.is_done = False
        frame.is_processing = False

    #override
    def on_result (self, host_dict, pf_sent, pf_result):
        print("on_result")
        frame = self.frames[pf_result.idx]
        print(f"pf_sent={pf_sent}, pf_result={pf_result} in InteractiveMergerSubprocessor")
        frame.is_processing = False
        print(f"if frame.cfg == pf_result.cfg: {frame.cfg}=={pf_result.cfg}")
        if frame.cfg == pf_result.cfg:
            frame.is_done = True
            frame.image = pf_result.image

    #override
    def get_data(self, host_dict):
        if self.is_interactive and self.is_interactive_quitting:
            return None

        for i in range ( min(len(self.frames_idxs), self.prefetch_frame_count) ):
            frame = self.frames[ self.frames_idxs[i] ]

            if not frame.is_done and not frame.is_processing and frame.cfg is not None:
                frame.is_processing = True
                return InteractiveMergerSubprocessor.ProcessingFrame(idx=frame.idx,
                                                           cfg=frame.cfg.copy(),
                                                           prev_temporal_frame_infos=frame.prev_temporal_frame_infos,
                                                           frame_info=frame.frame_info,
                                                           next_temporal_frame_infos=frame.next_temporal_frame_infos,
                                                           output_filepath=frame.output_filepath,
                                                           output_mask_filepath=frame.output_mask_filepath,
                                                           need_return_image=True )

        return None

    #override
    def get_result(self):
        return 0
