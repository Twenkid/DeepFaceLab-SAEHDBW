import os
import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
import cv2
import models
from core.interact import interact as io

#          elif key == ord('p') or (key >= ord('0') and key < (ord('9')+1)): #or key==ord('h') .. #22-4-2022 Twenkid -- for cyrillic default

#main
"""
                # HEAD
                head_lines = [
                    '[s]:save [b]:backup [enter]:exit',
                    '[p][0-9]:update [space]:next preview [l]:change history range', #0-9 Twenkid #22-4-2022
                    
   if key == ord('\n') or key == ord('\r'):
                s2c.put ( {'op': 'close'} )
            elif key == ord('s') or key == ord('2'):
                s2c.put ( {'op': 'save'} )
            elif key == ord('b') or key == ord('0'):
                s2c.put ( {'op': 'backup'} )
            elif key == ord('p') or (key >= ord('3') and key < (ord('8')+1)): #or
            #elif key == ord('p') or (key >= ord('0') and key < (ord('9')+1)): #or key==ord('h') .. #22-4-2022 Twenkid -- for cyrillic default
                if not is_waiting_preview::
            
"""       
user_dfl_save_interval_min = 25 #15 25

#if user_dfl_save_interval_min in os.environ: user_dfl_save_interval_min=int(os.environ["user_dfl_save_interval_min"]) #3-5-2022
if "user_dfl_save_interval_min" in os.environ: user_dfl_save_interval_min=int(os.environ["user_dfl_save_interval_min"]) #3-5-2022
print(f"Trainer.user_dfl_save_interval_min={user_dfl_save_interval_min}")

use_color_input_and_grayscale_model = "color_input_and_grayscale_model" in os.environ;   
forced_channels = 1 if use_color_input_and_grayscale_model else 3 #25-4-2022
#bDebugTrainer_update_preview = False #27-4-2022
debug_trainer_preview = False #Renamed, #1-5-2022
user_head_line_height = 16 #def.=15 27-4-2022
user_max_size = 800 #def. = 800
use_bw_input = "use_bw_input" in os.environ
print(f"cv2ex.bw_input = {use_bw_input}")

def trainerThread (s2c, c2s, e,
                    model_class_name = None,
                    saved_models_path = None,
                    training_data_src_path = None,
                    training_data_dst_path = None,
                    pretraining_data_path = None,
                    pretrained_model_path = None,
                    no_preview=False,
                    force_model_name=None,
                    force_gpu_idxs=None,
                    cpu_only=None,
                    silent_start=False,
                    execute_programs = None,
                    debug=False,
                    **kwargs):
    while True:
        try:
            start_time = time.time()

            save_interval_min = user_dfl_save_interval_min #15 # RAM DISK -- 60 #121 #241 #25 #editet 9.4.2022

            if not training_data_src_path.exists():
                training_data_src_path.mkdir(exist_ok=True, parents=True)

            if not training_data_dst_path.exists():
                training_data_dst_path.mkdir(exist_ok=True, parents=True)

            if not saved_models_path.exists():
                saved_models_path.mkdir(exist_ok=True, parents=True)
                            
            model = models.import_model(model_class_name)(
                        is_training=True,
                        saved_models_path=saved_models_path,
                        training_data_src_path=training_data_src_path,
                        training_data_dst_path=training_data_dst_path,
                        pretraining_data_path=pretraining_data_path,
                        pretrained_model_path=pretrained_model_path,
                        no_preview=no_preview,
                        force_model_name=force_model_name,
                        force_gpu_idxs=force_gpu_idxs,
                        cpu_only=cpu_only,
                        silent_start=silent_start,
                        debug=debug)

            is_reached_goal = model.is_reached_iter_goal()

            shared_state = { 'after_save' : False }
            loss_string = ""
            save_iter =  model.get_iter()
            def model_save():
                if not debug and not is_reached_goal:
                    io.log_info ("Saving....", end='\r')
                    model.save()
                    shared_state['after_save'] = True
                    
            def model_backup():
                if not debug and not is_reached_goal:
                    model.create_backup()             

            def send_preview():
                if not debug:
                    previews = model.get_previews()
                    c2s.put ( {'op':'show', 'previews': previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
                else:
                    previews = [( 'debug, press update for new', model.debug_one_iter())]
                    c2s.put ( {'op':'show', 'previews': previews} )
                e.set() #Set the GUI Thread as Ready

            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('Model already trained to target iteration. You can use preview.')
                else:
                    io.log_info('Starting. Target iteration: %d. Press "Enter" to stop training and save model.' % ( model.get_target_iter()  ) )
            else:
                io.log_info('Starting. Press "Enter" to stop training and save model.')

            last_save_time = time.time()

            execute_programs = [ [x[0], x[1], time.time() ] for x in execute_programs ]

            for i in itertools.count(0,1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if prog_time > 0 and (cur_time - start_time) >= prog_time:
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time)  >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("Unable to execute program: %s" % (prog) )

                    if not is_reached_goal:

                        if model.get_iter() == 0:
                            io.log_info("")
                            io.log_info("Trying to do the first iteration. If an error occurs, reduce the model parameters.")
                            io.log_info("")
                            
                            if sys.platform[0:3] == 'win':
                                io.log_info("!!!")
                                io.log_info("Windows 10 users IMPORTANT notice. You should set this setting in order to work correctly.")
                                io.log_info("https://i.imgur.com/B7cmDCB.jpg")
                                io.log_info("!!!")

                        iter, iter_time = model.train_one_iter()

                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")
                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s]".format ( time_str, iter, '{:0.4f}'.format(iter_time) )
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms]".format ( time_str, iter, int(iter_time*1000) )

                        if shared_state['after_save']:
                            shared_state['after_save'] = False
                            
                            mean_loss = np.mean ( loss_history[save_iter:iter], axis=0)

                            for loss_value in mean_loss:
                                loss_string += "[%.4f]" % (loss_value)

                            io.log_info (loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.4f]" % (loss_value)

                            if io.is_colab():
                                io.log_info ('\r' + loss_string, end='')
                            else:
                                io.log_info (loss_string, end='\r')

                        if model.get_iter() == 1:
                            model_save()

                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info ('Reached target iteration.')
                            model_save()
                            is_reached_goal = True
                            io.log_info ('You can use preview now.')
                
                need_save = False
                while time.time() - last_save_time >= save_interval_min*60:
                    last_save_time += save_interval_min*60
                    need_save = True
                
                if not is_reached_goal and need_save:
                    model_save()
                    send_preview()

                if i==0:
                    if is_reached_goal:
                        model.pass_one_iter()
                    send_preview()

                if debug:
                    time.sleep(0.005)

                while not s2c.empty():
                    input = s2c.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'backup':
                        model_backup()
                    elif op == 'preview':
                        if is_reached_goal:
                            model.pass_one_iter()
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break

                if i == -1:
                    break



            model.finalize()

        except Exception as e:
            print ('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    c2s.put ( {'op':'close'} )



def main(**kwargs):
    io.log_info ("Running trainer.\r\n")

    no_preview = kwargs.get('no_preview', False)

    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e), kwargs=kwargs )
    thread.start()

    e.wait() #Wait for inital load to occur.

    if no_preview:
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'close':
                    break
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )
    else:
        #wnd_name = "Training preview"
        wnd_name = "Training preview: " + kwargs['model_class_name'] + ": " + str(kwargs['saved_models_path']) #29-4-2022
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0
        iter = 0
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iter = input['iter'] if 'iter' in input.keys() else 0
                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            if debug_trainer_preview: print(f"preview_rgb.shape = {preview_rgb.shape}") #24-4-2022
                            (h, w, c) = preview_rgb.shape                            
                            max_h = max (max_h, h)
                            max_w = max (max_w, w)

                        #max_size = 800
                        max_size = user_max_size
                        if max_h > max_size:
                            max_w = int( max_w / (max_h / max_size) )
                            max_h = max_size

                        #make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append ( (preview_name, cv2.resize(preview_rgb, (max_w, max_h))) )
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False

                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                if len(selected_preview_rgb.shape)==3:
                  (h,w,c) = selected_preview_rgb.shape                  
                else: selected_preview_rgb=selected_preview_rgb[:,:,np.newaxis]  #BW ... #24-4-2022  #27-4-2022 --> new axis
                #(h,w) = selected_preview_rgb.shape; c = 1;   
                (h,w,c) = selected_preview_rgb.shape #; c = 1;   

                # HEAD
                head_lines = [
                    '[s|2]:save [b|0]:backup [enter]:exit [p|3-9]:update',
                    #'[p][0:9]:update[space]:next preview [l]:change history range', #0-9 Twenkid #22-4-2022
                    '[space]:next preview [l 1]:change history range',
                    'Preview: "%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
                    ]
                head_line_height = user_head_line_height #head_line_height = 15
                head_height = len(head_lines) * head_line_height
                #print(f"update_preview: (h,w,c)={h},{w},{c}")                
                head = np.ones ( (head_height,w,c) ) * 0.1                
                               
                for i in range(0, len(head_lines)):
                    t = i*head_line_height
                    b = (i+1)*head_line_height                    
                    #It's the same? #27-4-2022, 10:39
                    #if c == 1: print("head[... get_text_image...[0]"); head[t:b, 0:w] += imagelib.get_text_image (  (head_line_height,w,c) , head_lines[i], color=[0.8]*c )[0] #1 channel
                    #else: head[t:b, 0:w] += imagelib.get_text_image (  (head_line_height,w,c) , head_lines[i], color=[0.8]*c )                    
                    #head[t:b, 0:w] += imagelib.get_text_image ((head_line_height,w,c) , head_lines[i], color=[0.8]*c)
                    head[t:b, 0:w] += imagelib.get_text_image ((head_line_height,w,c) , head_lines[i])
                    if debug_trainer_preview: 
                      print(f"(head[{t}:{b}, 0:{w}] += imagelib.get_text_image(({head_line_height},{w},{c}), {head_lines[i]}, color={[0.8]*c}")
                      print("Leave default color (1,1,1)")
                    
                final = head
                if debug_trainer_preview: print(f"np.sum(head) = {np.sum(head)}")

                if loss_history is not None:
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]

                    
                    #lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)                    
                    lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c) #forced_channels) #25-4-2022
                    final = np.concatenate ( [final, lh_img], axis=0 )

                if debug_trainer_preview: print(f"final.shape={final.shape}, selected_preview_rgb.shape = {selected_preview_rgb.shape}")
                #if use_color_input_and_grayscale_model: final = np.concatenate ( [final[0], selected_preview_rgb], axis=0 )
                if use_color_input_and_grayscale_model or use_bw_input:  #or ... #7-5-2022
                  selected_preview_rgb = np.array(selected_preview_rgb)
                  if debug_trainer_preview:
                    print(f"final.shape={final.shape}, selected_preview_rgb.shape = np.array(sel...) = {selected_preview_rgb.shape}")
                  #final = np.concatenate ( [final, selected_preview_rgb], axis=0)
                  #no ?why no? final = np.concatenate ( [final[:,:,0], selected_preview_rgb], axis=0) #? 25-4-2022
                  #no final,_,_ = np.moveaxis(final,-1,0)
                  
                  #ff = final[:,:,0]
                  #print(f"final[:,:,0].shape = {ff}")
                  
                  """ 
                  if len(final.shape)!=len(selected_preview_rgb.shape): 
                     print("if len(final.shape)!=len(selected_preview_rgb.shape):")
                     final = np.concatenate ( [final[:,:,0], selected_preview_rgb], axis=0) #? 25-4-2022                                       
                  else: final = np.concatenate ( [final, selected_preview_rgb], axis=0)
                  """
                  #BLACKWHITE
                  if debug_trainer_preview: print(f"selected_preview_rgb.reshape --> { selected_preview_rgb.shape}")
                  
                  #selected_preview_rgb.reshape((selected_preview_rgb.shape[0], selected_preview_rgb.shape[1], 1)) #27-4-2022
                  if (len(selected_preview_rgb.shape)==2):
                    selected_preview_rgb=selected_preview_rgb[:,:,np.newaxis] #27-4-2022 --> better add the axis above
                    if debug_trainer_preview: print(f"final.shape = {final.shape}, selected_preview_rgb.reshape/selected_preview_rgb[:,:,np.newaxis] --> { selected_preview_rgb.shape}")
                  final = np.concatenate ( [final, selected_preview_rgb], axis=0)
                  
                  #print(f"np.moveaxis:final.shape={final.shape}, selected_preview_rgb.shape = np.array(sel...) = {selected_preview_rgb.shape}")
                else: final = np.concatenate ( [final, selected_preview_rgb], axis=0)                
                final = np.clip(final, 0, 1)
                io.show_image( wnd_name, (final*255).astype(np.uint8) )
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == ord('\n') or key == ord('\r'):
                s2c.put ( {'op': 'close'} )
            elif key == ord('s') or key == ord('2'):
                s2c.put ( {'op': 'save'} )
            elif key == ord('b') or key == ord('0'):
                s2c.put ( {'op': 'backup'} )
            elif key == ord('p') or (key >= ord('3') and key < (ord('8')+1)): #or
            #elif key == ord('p') or (key >= ord('0') and key < (ord('9')+1)): #or key==ord('h') .. #22-4-2022 Twenkid -- for cyrillic default
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put ( {'op': 'preview'} )
            elif key == ord('l') or key == ord('1'):
                if show_last_history_iters_count == 0:      
                   show_last_history_iters_count = 1000
                if show_last_history_iters_count == 1000:
                   show_last_history_iters_count = 2500
                if show_last_history_iters_count == 2500:
                   show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                   show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                   show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                   show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                   show_last_history_iters_count = 0
                update_preview = True
            #elif key == ord('i'):
            #  if "print_samples_info" in os.environ:
            #     os.environ["print_samples_info"]=not os.environ["print_samples_info"]
            #     update_preview = True             
            elif key == ord(' '): #when selecting the first preview, space doesn't show the masks, but shows another set of images!
                selected_preview = (selected_preview + 1) % len(previews)
                #print(f"selected_preview = {selected_preview} of {len(previews)}")
                update_preview = True

            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )

        io.destroy_all_windows()