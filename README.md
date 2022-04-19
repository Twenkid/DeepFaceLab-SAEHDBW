# DeepFaceLab

Notes, experience, tools, deepfakes

19.4.2022

```
preview_period = 334
preview_period_colab = 200

C:\DFL\DeepFaceLab_DirectX12\_internal\DeepFaceLab\models\Model_SAEHD\Model.py
  #override
    def should_save_preview_history(self):
       return (not io.is_colab() and self.iter % preview_period == 0) or (io.is_colab() and self.iter % preview_period_colab == 0)
    #    return (not io.is_colab() and self.iter % ( 10*(max(1,self.resolution // 64)) ) == 0) or \
    #           (io.is_colab() and self.iter % 100 == 0)

Also in ModelBase.py

Similar function, ... but it's overriden in the separate models.
```


VS: copy with line numbers 


17.4.2022

Reduce noise, training at home/PC in the bedroom:
Намаляване на шума при обучение вкъщи на компютър в спалнята (през нощта):

* Win 10, Power Options/Опции на захранването: Икономичен режим, до 40-50% на ЦПУ - спира да бучи циклично (Core i5 6500 3.3 GHz).

DFL Colab - от оригиналния iperov ... e... drive - edge --> 


9.4.2022

* https://mrdeepfakes.com/forums/thread-guide-deepfacelab-2-0-guide#
* https://github.com/iperov/DeepFaceLive
* https://github.com/iperov/DeepFaceLab

Try also Live

* If downloading the github repo and running the main.py as with Colab, it will run only on CPU. It must be built.
* If using the CPU version and newer OpenCV than 4.1, edit:
``` 
C:\Deepface\core\imagelib\warp.py

Locate:

def gen_warp_params ...

 random_transform_mat = cv2.getRotationMatrix2D((int(w // 2), int(w // 2)), rotation, scale) 
  random_transform_mat[:, 2] += (tx*w, ty*w)

#### (int(w // 2), int(w // 2)) #####
```

* Download the Builds and run/edit the .bat files
 
* After install, find trainer.py 
Change trainer thread: 

```
 while True:
        try:
            start_time = time.time()

            save_interval_min = 241 # 25  -- or whatever

```

To whatever you like.

* SAEHD 192

```
Running trainer.

Choose one of saved models, or enter a name to create a new model.
[r] : rename
[d] : delete

[0] : sae-192 - latest
[1] : ksaehd
 : 0
0
Loading sae-192_SAEHD model...

Choose one or several GPU idxs (separated by comma).

[CPU] : CPU
  [0] : NVIDIA GeForce GTX 750 Ti
  [1] : Intel(R) HD Graphics 530

[1] Which GPU indexes to choose? : 0
0

Press enter in 2 seconds to override model settings.
[0] Autobackup every N hour ( 0..24 ?:help ) :
0
[y] Write preview history ( y/n ?:help ) :
y
[n] Choose image for the preview history ( y/n ) : y
[3001] Target iteration : 10001
10001
[n] Flip SRC faces randomly ( y/n ?:help ) :
n
[y] Flip DST faces randomly ( y/n ?:help ) :
y
[4] Batch_size ( ?:help ) :
4
[y] Masked training ( y/n ?:help ) :
y
[y] Eyes and mouth priority ( y/n ?:help ) :
y
[n] Uniform yaw distribution of samples ( y/n ?:help ) :
n
[n] Blur out mask ( y/n ?:help ) :
n
[n] Place models and optimizer on GPU ( y/n ?:help ) :
n
[n] Use AdaBelief optimizer? ( y/n ?:help ) :
n
[n] Use learning rate dropout ( n/y/cpu ?:help ) :
n
[y] Enable random warp of samples ( y/n ?:help ) :
y
[0.0] Random hue/saturation/light intensity ( 0.0 .. 0.3 ?:help ) :
0.0
[0.0] GAN power ( 0.0 .. 5.0 ?:help ) :
0.0
[0.0] Face style power ( 0.0..100.0 ?:help ) :
0.0
[0.0] Background style power ( 0.0..100.0 ?:help ) :
0.0
[rct] Color transfer for src faceset ( none/rct/lct/mkl/idt/sot ?:help ) :
rct
[n] Enable gradient clipping ( y/n ?:help ) :
n
[y] Enable pretraining mode ( y/n ?:help ) :
y
Initializing models: 100%|###############################################################| 5/5 [00:01<00:00,  2.56it/s]
Loaded 15843 packed faces from C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces
Sort by yaw: 100%|##################################################################| 128/128 [00:00<00:00, 239.71it/s]
Sort by yaw: 100%|##################################################################| 128/128 [00:00<00:00, 238.36it/s]
Choose image for the preview history. [p] - next. [space] - switch preview type. [enter] - confirm.
=================== Model Summary ====================
==                                                  ==
==            Model name: sae-192_SAEHD             ==
==                                                  ==
==     Current iteration: 3001                      ==
==                                                  ==
==----------------- Model Options ------------------==
==                                                  ==
==            resolution: 192                       ==
==             face_type: wf                        ==
==     models_opt_on_gpu: False                     ==
==                 archi: liae-ud                   ==
==               ae_dims: 128                       ==
==                e_dims: 96                        ==
==                d_dims: 64                        ==
==           d_mask_dims: 16                        ==
==       masked_training: True                      ==
==       eyes_mouth_prio: True                      ==
==           uniform_yaw: True                      ==
==         blur_out_mask: False                     ==
==             adabelief: False                     ==
==            lr_dropout: n                         ==
==           random_warp: False                     ==
==      random_hsv_power: 0.0                       ==
==       true_face_power: 0.0                       ==
==      face_style_power: 0.0                       ==
==        bg_style_power: 0.0                       ==
==               ct_mode: rct                       ==
==              clipgrad: False                     ==
==              pretrain: True                      ==
==       autobackup_hour: 0                         ==
== write_preview_history: True                      ==
==           target_iter: 10001                     ==
==       random_src_flip: False                     ==
==       random_dst_flip: True                      ==
==            batch_size: 4                         ==
==             gan_power: 0.0                       ==
==        gan_patch_size: 16                        ==
==              gan_dims: 16                        ==
==                                                  ==
==------------------- Running On -------------------==
==                                                  ==
==          Device index: 0                         ==
==                  Name: NVIDIA GeForce GTX 750 Ti ==
==                  VRAM: 1.45GB                    ==
==                                                  ==
======================================================
Starting. Target iteration: 10001. Press "Enter" to stop training and save model.
[
[11:54:23][#003094][1766ms][2.5891][1.9999]

Pretraining

```
