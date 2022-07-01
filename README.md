# DeepFaceLab - New Grayscale SAEHDBW Model and Code Review / Documentation of the source files

## Notes, experience, tools, deepfakes

### Manual

<a href="https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/Manual.md">https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/Manual.md</a>


22.6.2022

Premiere! 

![image](https://user-images.githubusercontent.com/23367640/175982252-3d79a921-6261-4fc3-ad56-8737b812b955.png)

https://artificial-mind.blogspot.com/2022/06/arnold-schwarzenegger-governor-of.html

https://www.youtube.com/watch?v=n2JMGqxOYfA&feature=youtu.be

Created with DFL-SAEHDBW, df-udt, 192x192, 128x48x32x16. Trained on a Geforce 750 Ti 2 GB on Windows 10.

![image](https://user-images.githubusercontent.com/23367640/175983477-be704259-ae9b-41ec-a245-7ea30af6c516.png)

![image](https://user-images.githubusercontent.com/23367640/175983001-12bd3314-6119-4923-a7f3-e63707bb7427.png)


![image](https://user-images.githubusercontent.com/23367640/175983775-d8f1ae00-16d0-497f-817f-68ed7dc51204.png)

![image](https://user-images.githubusercontent.com/23367640/175986539-bec8c060-828d-41d1-8b5f-a81695213b89.png)
![image](https://user-images.githubusercontent.com/23367640/175986660-bdf0e5d9-59e7-4e02-a94c-8a0c7f83bc19.png)



![image](https://user-images.githubusercontent.com/23367640/175986888-d93e7375-a72d-4495-8785-b57870e136cb.png)

![image](https://user-images.githubusercontent.com/23367640/175986915-936591e3-d475-43fe-9ed6-5bc430a327f2.png)





~ 22.4.2022 -- Minor iterface changes (more keys for save, save preview periods and auto saving;later: possible forcing generation of new previews instead of keeping the same for the whole training etc.; reviewing the code

~ 25.4.2022? --> Started working on SAEHDBW - Grayscale deepfake model; research, experiments, modifications of the channel dimensions, studying the NN model.

## Goals of the project:
* Allow training of more "respectable" resolution models even on 2 GB GPUs, GeForce 750 Ti in particular, and on integrated GPUs
* Achieve several times? higher performance: either smaller models, higher resolution and/or more detailed models, although grayscale. 
* Study the code, if possible modify the architectures and optimize more: simplify/reduce the depth of some networks to check if they would achieve similar quality due to the single channel with improved performance. 

* ~ 27.4.2022 --> SAEHDBW - SUCCESS!

First correctly training version (last error fixed masks getting bug after untested change, numpy).
Initial mode: training from color input which is converted to grayscale during reading.
Now the model can train on 224x224 and 192x192 images on a 2 GB Geforce 750 Ti. (The quality etc. is to be checked at various AE dims, encoder dims, decoder dims.)


**Note:** _Initially working with the DirectX12 build with tensorflow-directml due to the CUDA version which didn't run (that issue was resolved later on 10.5.2022). The CPU is Core i5 6500. Iteration time varies and depends also on the CPU power mode (Economical/Balanced and their details) and the overclocking of the GPU. Initially I didn't use additional overclock (boost clock was up to 1176 MHz), from some moment I started using MSI Afterburner which allowed going above 1415 MHz for core clock and above 2900 MHz for RAM and that was not the maximum, but it doesn't sustain it all the time due to temperature and power limits set for safety and it may get unstable and interrupt, while the gain is small. When the batch size is bigger or there is heavy CPU processing, e.g. some color transfer modes such as SOT, the CPU load is higher._


liae-ud-r192-ae-96-48-48-12-bw_SAEHDBW  x f, ~900ms (R192...-ae-64-48-48 - almost the same it time ~ 860 ms)

R224-AE64-48-48-12-BW_SAEHDBW x f liae-ud ~ 1500-1600ms

Training on converted to grayscale pretrain faceset, resized to 384x384 (from 768x768). Checking how much details would be captured with different dimensions.

2.5.2022:

DF-UDT-256-96-32-32-16_SAEHDBW --> batch 4: ~ 1200 ms, (~ 1150 ms slightly more overclock); batch 5 ~ 1500 ms (OOM errors occasionaly)

Some model sizes on disk and batch size (for 750 Ti/2 GB)

```
Model Sizes: MB

LIAE-UD

liae-ud-r192-ae-96-48-48-12-bw_SAEHDBW -- 362 MB  (12 is == 16 mask dim)

R192-AE80-48-48-16-LIAE-UD-SAEHD-BW-PRETR_SAEHDBW-- 315 M
R192-AE64-48-48-16-LIAE-UD-SAEHDBW-PRETR_SAEHDBW  -- 269 M

liae-ud-r96-64-64-22_SAEHDBW -- 313 M
R224-AE64-48-48-12-BW_SAEHDBW -- 297 M  (12 is == 16)

liae-ud-r96-24-24-12_SAEHDBW -- 45.6 M (12 is == 16)
liae-ud-r96-32-32-12-bw_SAEHDBW -- 96 M (12 is == 16)

LIAE-UDT-R128-96-32-32-16_SAEHDBW -- 209 M  B: 4,6,8 (B=8: it= 444-463 ms (530, Lower power mode) --> ~4K@4, 13K@6 --> 8), 4.5.2022 --> train at f (also do on mf)

LIAE-UDT-192-128-32-32-16-SAEHDBW_SAEHDBW_summary -- 270 MB
LIAE_UDT-192-96-32-48-16-SAEHDBW_SAEHDBW_summary -- 346 MB ==> would it be beneficial if having a lower dim. encoder than decoder (yet more parameters overall and thus more detail?)

LIAE-UDT 192-96-32-48-16 vs LIAE-UD 192-96-48-48-16 ?


"G:\SAEHDBW\liae-udt-192-96-32-32-16-SAEHDBW_SAEHDBW... - 234 МБ

"G:\SAEHDBW\liae-ud-192-128-32-32-16_SAEHDBW_src_dst_opt.npy" - 273 МБ

*** DF-UD ***

dfud-r96-32-32-12-bw_SAEHDBW_summary.txt -- 104 M
DF-UDT-256-96-32-32-16_SAEHDBW -- 281 M B: 4, 5 (OOM in minutes sometimes)
DF-UDT-R96-64-24-24-16-SAEHDBW_SAEHDBW -- 50 MB , train @mf 

"G:\SAEHDBW\df-udt-192-96-32-32-16_SAEHDBW - 285 МБ

"G:\SAEHDBW\df-udt-192-128-48-32-16_SAEHDBW - 345 МБ

df-ud-192-128-48-48-16_SAEHDBW_summary - 345 МБ

(Check the quality of df-ud and df-udt with the same number of params - if there's enough patience to train them. Does 48-32 is good enough, varying number of channels/dimensions for the encoder and decoder? Encoder > decoder ... Also: 128/32/32? Mapping the default 256/64/64 for color 128 pix)

== Какво ще е качеството? Сравнимо ли ще е? Доколко няма да достигнат размерностите?

df-udt-192-80-32-32-16_SAEHDBW -- 260 MB, batch 8 speed ~ or faster than  df-udt-192-128-48-32-16_SAEHDBW @  batch 6  #13-5-2022, ~21h;


```

Trying 't', searching for higher sharpness; various settings tried.

* Only 1.45 GB available. Connecting the monitor to the integrated GPU, but OS still reserves that amount and sometimes even with just 77% usage when connecting a monitor to the integrated GPU, after trying to create a bigger batch the model doesn't start training. (Windows 10 issue.)

* 10-5-2022 - Debugged the CUDA build so now I can use it. (I used the DirectX12 build so far, because the CUDA-one hanged with no output). The solution provided 33% speed-up! https://github.com/iperov/DeepFaceLab/issues/5515 "device_lib.list_local_devices() doesn't return in the CUDA build up to 2080 #5515"
* 
* 11.5.2022 - After a series of GPU-related crashes when trying to run big models at the edge in the CUDA build, with sizes which previously were training in the DirectX12 version, it seems that the DX12 version, i.e. tensorflow-directml uses less memory than CUDA. It is possible to train: DF-UDT-256-96-32-32-16_SAEHDBW --> as recorded recently: batch 4: ~ 1200 ms, (~ 1150 ms slightly more overclock); batch 5 ~ 1500 ms (OOM errors occasionaly).

## Sample pretraining and training experiments

* SAEHDBW df-udt-mf-R192-128-48-32-16, batch 6 pretraining on the custom subset of the built-in faceset.pak, reextracted* at 384x384 grayscale and with removed many images, lastly about 14634. It still has a few "bad" samples with babies and hand in the mouth, some musician with an instrument etc, hair covering some of the eye etc. Portraits with glasses are kept, except extremely strange ones. Microphones, hands and other objects crossing the face are removed, expect a few and when it is slightly touching; etc.

* 12.5.2022 Note: They had better be resized instead of reextraceted, using a modified DFL script for resizing, but I haven't reached that part of the code then.

See the process with more examples etc.:  <a href="https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/Pretraining-df-udt-mf-192-128-48-32-16.md">https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/Pretraining-df-udt-mf-192-128-48-32-16.md</a>

**441K-** 

Eyelashes

![image](https://user-images.githubusercontent.com/23367640/168331381-0c9123c7-cc2b-46e6-8efe-f29e8e05547e.png)
![image](https://user-images.githubusercontent.com/23367640/168331557-ba4ba72e-7621-4934-8758-24c523fd2bf1.png)
![image](https://user-images.githubusercontent.com/23367640/168331593-54207f0f-3569-4be9-9174-676a65c913cb.png)
![image](https://user-images.githubusercontent.com/23367640/168331611-acabaab0-fbb8-4ce6-968a-b552331a4768.png)

Teeth

![image](https://user-images.githubusercontent.com/23367640/168331982-671b7f6b-ba16-4037-8c69-d69e097d269d.png)



* SAEHDBW liae-ud-r96-32-32-16; no pretraining; mouth and eye priority

Iterations: about 220-230 ms for batch size 8 and about 320-330 for 12 after 370K, training from color images. This model is still progressing. The faceset is currently about 400 images for Biden and 2200 images for the other person (K.P.), where the Biden's are of a higher quality and sharper. Almost all of K's images are from videos and low resolution. Initially training only on about 200 stills of Biden, then about 200 frames from a clip of a lower quality. The model has small dimensions so very high quality can't be captured anyway.

![image](https://user-images.githubusercontent.com/23367640/166139358-17555ce6-06b0-4089-98be-41a2d596e34f.png)

* Modified faceset - reduced to about 14600 images, removed many which I didn't "like", having occlusions - microphones, hands etc., grayscale 384x384



* SAEHDBW liae-udt-r128-96-32-32-16 Pretraining; mouth and eye priority

![image](https://user-images.githubusercontent.com/23367640/167268956-ed171f28-8132-4762-ad9b-daf48c11ee17.png)
![image](https://user-images.githubusercontent.com/23367640/167268964-7154019d-46a4-4629-bd34-de72f65b4466.png)
![image](https://user-images.githubusercontent.com/23367640/167268973-dc30dad3-d080-4c3f-b21b-fbb6fd8793b1.png)
![image](https://user-images.githubusercontent.com/23367640/167268983-7240d846-ec19-4aa9-a125-d0a35aabf9d0.png)
![image](https://user-images.githubusercontent.com/23367640/167268986-50996c10-6542-46fa-b712-a9a1b4464625.png)

...


Continuing working on the project.

Training with grayscale input (8-bit jpg, png) significant improvement (twice) for the pre-training dataset color images.
Modify Extract to extract to grayscale.
Unpacking the pretrain faceset, 768x768 color. Extract to 384x384 BW.

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


======



"G:\SAEHDBW\liae-udt-192-96-32-32-16-SAEHDBW_SAEHDBW_decoder.npy"
"G:\SAEHDBW\liae-udt-192-96-32-32-16-SAEHDBW_SAEHDBW_inter_B.npy"
"G:\SAEHDBW\liae-udt-192-96-32-32-16-SAEHDBW_SAEHDBW_encoder.npy"
"G:\SAEHDBW\liae-udt-192-96-32-32-16-SAEHDBW_SAEHDBW_inter_AB.npy"
"G:\SAEHDBW\liae-udt-192-96-32-32-16-SAEHDBW_SAEHDBW_data.dat"
"G:\SAEHDBW\liae-udt-192-96-32-32-16-SAEHDBW_SAEHDBW_src_dst_opt.npy"
234 МБ

"G:\SAEHDBW\liae-ud-192-128-32-32-16_SAEHDBW_src_dst_opt.npy"
"G:\SAEHDBW\liae-ud-192-128-32-32-16_SAEHDBW_encoder.npy"
"G:\SAEHDBW\liae-ud-192-128-32-32-16_SAEHDBW_data.dat"
"G:\SAEHDBW\liae-ud-192-128-32-32-16_SAEHDBW_decoder.npy"
"G:\SAEHDBW\liae-ud-192-128-32-32-16_SAEHDBW_inter_AB.npy"
"G:\SAEHDBW\liae-ud-192-128-32-32-16_SAEHDBW_inter_B.npy"
273 МБ



*** DF-UD ***

dfud-r96-32-32-12-bw_SAEHDBW_summary.txt -- 104 M
DF-UDT-256-96-32-32-16_SAEHDBW -- 281 M B: 4, 5 (OOM in minutes sometimes)
DF-UDT-R96-64-24-24-16-SAEHDBW_SAEHDBW -- 50 MB , train @mf 

"G:\SAEHDBW\df-udt-192-96-32-32-16_SAEHDBW_inter.npy"
"G:\SAEHDBW\df-udt-192-96-32-32-16_SAEHDBW_encoder.npy"
"G:\SAEHDBW\df-udt-192-96-32-32-16_SAEHDBW_data.dat"
"G:\SAEHDBW\df-udt-192-96-32-32-16_SAEHDBW_src_dst_opt.npy"
"G:\SAEHDBW\df-udt-192-96-32-32-16_SAEHDBW_decoder_dst.npy"
"G:\SAEHDBW\df-udt-192-96-32-32-16_SAEHDBW_decoder_src.npy"
285 МБ

"G:\SAEHDBW\ df-udt-192-128-48-32-16_SAEHDBW_decoder_src.npy"
"G:\SAEHDBW\ df-udt-192-128-48-32-16_SAEHDBW_encoder.npy"
"G:\SAEHDBW\ df-udt-192-128-48-32-16_SAEHDBW_inter.npy"
"G:\SAEHDBW\ df-udt-192-128-48-32-16_SAEHDBW_summary.txt"
"G:\SAEHDBW\ df-udt-192-128-48-32-16_SAEHDBW_data.dat"
"G:\SAEHDBW\ df-udt-192-128-48-32-16_SAEHDBW_src_dst_opt.npy"
"G:\SAEHDBW\ df-udt-192-128-48-32-16_SAEHDBW_decoder_dst.npy"
345 МБ

df-ud-192-128-48-48-16_SAEHDBW_summary
345 МБ
