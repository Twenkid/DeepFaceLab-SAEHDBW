# DeepFaceLab-SAEHDBW or **Arnoldifier, ArnoldDFR**
## New Grayscale SAEHDBW Model for higher performance, then Colorization of the result; Integration with Wav2Lip etc. and Code Review / Documentation of the source files (future work)

![image](https://user-images.githubusercontent.com/23367640/175982252-3d79a921-6261-4fc3-ad56-8737b812b955.png)

## Notes, experience, tools, deepfakes

Some of the points are goals, TBD.

**News**

2.4.2023: Episode 0: The Pilot experimental version of Episode 1 with an earlier version of the model and the workflow, completed in May 2022, but not released up to now: https://youtu.be/2CMmd494Dqw
![image](https://user-images.githubusercontent.com/23367640/229377472-916232a7-c271-4af5-9873-a6bd4eb5734f.png)

23.3.2023: One of the scripts for colorization is added: the one that trains a pix2pix model: grayscale to color faces.
I didn't upload it when I made it, because the workflow was a bit laborious, but I didn't have energy to make it easier for users back then.

https://github.com/Twenkid/DeepFaceLab-SAEHDBW/tree/main/DeepFaceLab_DirectX12/_internal/DeepFaceLab/colorize

The other pieces and an example to be added later. See the colorized example below.

21.2.2023: Recently I trained a new model to Arnold (retrained the Kiril model), an old idea to "puppet" the face myself, although the source wasn't specially "Arnold-related". It achieved very low error. There was an unknown hanging of the CUDA build during the initialization phase (I have fixed one bug of that kind in May last year), which I skipped fixing and used the DirectX version instead, up to today - after a bit of LRD training for two hours or so and managed to fix it - updating a few files. (I don't know whether that error is present in any other setups though, I haven't tested ArnoldDFR on other machines and nobody has reported, although there are 10 stars and 3 forks so far).
DeepFaceLab_SAEHDBW_22_5_2022\core\leras\nn.py
DeepFaceLab_SAEHDBW_22_5_2022\core\leras\device.py
Sample training file:

An issue in DeepfaceLab repo about this project (2 July 2022): https://github.com/iperov/DeepFaceLab/issues/5535

### Manual

<a href="https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/Manual.md">https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/Manual.md</a>

Watch the series: "Arnold Scwarzenegger: The Governor of Bulgaria"</a>

**Premiere!  Part I**

![image](https://user-images.githubusercontent.com/23367640/175982252-3d79a921-6261-4fc3-ad56-8737b812b955.png)

https://artificial-mind.blogspot.com/2022/06/arnold-schwarzenegger-governor-of.html

https://www.youtube.com/watch?v=n2JMGqxOYfA&feature=youtu.be

* **Part II**: **"Arnold meets his illegal daughter Lena Borislavova Scwarzenegger"** https://youtu.be/k1MZ8eIUjXE

![image](https://user-images.githubusercontent.com/23367640/178838680-44e7398c-a0b4-4548-9c82-9a2d3430886c.png)


* **Part III**: Combined with Wav2Lip - the mouth is synchronized; no finetuning; another pass of DFL to repair the artifacts (however there is some loss of contrast and sharpness)  https://youtu.be/4F7PB7wBEXk

![image](https://user-images.githubusercontent.com/23367640/179422086-2af2a887-4681-4f58-a68f-d16f4f6a04df.png)



* **Part IV**: Lip-syncing with finetuning repair and synthesized cloned voices (RealTimeVoiceCloning) over original script ("full directing") with Arnold, Lena Schwarzenegger and Stoltenberg    https://youtu.be/X56QkNzkkVM

![image](https://user-images.githubusercontent.com/23367640/196010523-9bf7514c-a206-4ee7-8cfd-4aa20cc2a4cc.png)

 
* **Part V**: 
Arnold's superfood and discussing Bulk & Cut and being shredded all the time: https://youtu.be/40VvW07aqdM

![image](https://user-images.githubusercontent.com/23367640/219415026-d370765c-a44a-4388-a603-862b6ff1fef5.png)

Суперхраната на Арнолд Шварценегер: Губернаторът на България - Част 5

"Produced with Arnoldifier, a modified  Deepfacelab which I made to train grayscale models with 3x higher performance/quality. This is trained on a GF 750 Ti 2 GB."
#deepfake #arnold #twenkid #fitness #фитнес #културизъм #bodybuilding #Arnoldifier #deepfacelab


Latest news:

* 30.9.2022 - Trained LIAE-UD 192-96-32-32-16, 246 MB

I wanted to see whether 96 dimensions of the autoencoder would be enough and it happened that they actually were. I pretrained on a modified DFL faceset to 500-some K it., then trained up to 256K Kiril-to-Arnold, uniform YAW - so far I didn't use it and that was a mistake. Now the profiles develop fine since the early stages. Due to another silly crash and lack of backup though (LOL) now I'm training again from a backup of the pretrained model from about 300K. Let's see whether the 200K-2xxK additional iterations of the pretraining contribute, that could be saved in the future.

* Plans for improvements: "Profile-fixer" stage for bad borders of semi-profile and profile poses. It seems that the masks have to be convex and for profiles they are always bad and cut a big chunk in front of the forehead and the nose which results in contours attached to the face. Sometimes the recognized face/mask is smaller than the target and a "ghost" of the original nose etc. appear.


* 4.10.2022 - Completed Kiril-to-Arnold LIAE-UD 192-96-32-32-16

One new "discovery": the model was trained with "Uniform Yaw" turned on, which lead to more balance and faster convergence of the profile faces.
The pretraining for ~300K was enough, then about 300-347K (with final tens K it. with LRD). I didn't trained CT (usually I did sot-m for some 10Ks it. with LRD).
It is slow, on the CPU. I just merged with sot-m and it seems OK (not perfect, but that rather needs additional layer of processing which is future work). So the slow color transfer mode seems to be avoidable with acceptable quality, and note that the test dataset is diverse, it is not a single clip from an interview etc., especially Arnold's set.

Interestingly, the smallest model so far, 246 MB displays higher quality and more realistic output than some 345 MB models (df-ud-t 192-128-48-32-16, maybe due to the asymmetrical encoder-decoder). I noticed a little "lighting up" of the nose etc. in one of the sequences (Parliament background), possibly the previous model (df-ud 256-128-32-32) was more stable, but I need to check it out and make a systematic comparison.

**Note that the default dimensions for color DF and resolution of just 128x128 in DFL are 256-64-64-22.**

Below: no additional sharpening applied

![image](https://user-images.githubusercontent.com/23367640/193997352-28ee21f5-7235-422e-83d9-319afbe46dfe.png)

![image](https://user-images.githubusercontent.com/23367640/193997372-c46a3d5e-0a00-4562-8334-eda14fe308d9.png)

![image](https://user-images.githubusercontent.com/23367640/193997524-a3ffe6fb-5d03-496e-9f8a-7f27423ebb39.png)

![image](https://user-images.githubusercontent.com/23367640/194002377-bf856a6d-a036-4a59-89ac-72841d2318ed.png)


* Sharpening, whole image (23, IrfanView)

![image](https://user-images.githubusercontent.com/23367640/193997774-df8a994f-b028-46e7-a604-ae23701d3fae.png)
![image](https://user-images.githubusercontent.com/23367640/194002523-b1042c8d-4245-446f-bce5-23085bf951d5.png)
![image](https://user-images.githubusercontent.com/23367640/194002553-55102d26-2e23-473f-850d-8bf7e8f7bea4.png)

Good profiles, except a glitch in the masks which sometimes leaves a trailing contour, the new face doesn't cover
the original face etc., the mask is convex and doesn't follow the shape of the face. I have ideas for correction: future work.

![image](https://user-images.githubusercontent.com/23367640/194002895-49894625-25e3-440c-89ee-6b2a75169750.png)

![image](https://user-images.githubusercontent.com/23367640/194003946-6c565517-2056-4c4c-bb88-f0899fb513ad.png)

![image](https://user-images.githubusercontent.com/23367640/194004160-4eb359b8-444b-4c56-81f5-5f40e9be0164.png)

...

![image](https://user-images.githubusercontent.com/23367640/193997901-1f67c1ba-883f-484a-8c47-8b6243814ca3.png)
![image](https://user-images.githubusercontent.com/23367640/193997960-24790f1a-5b97-41ad-a4d8-1afd3f3d6d7b.png)
![image](https://user-images.githubusercontent.com/23367640/193998130-1638dcfb-4604-41ec-ba0d-9bc8e4668822.png)

![image](https://user-images.githubusercontent.com/23367640/194003325-fd9a9960-e512-4ddc-b494-881a2756cf06.png)

* Sharpening: 32, IrfanView - FullHD image. Note that the Video source has wrong focus: the background instead of the character.

![image](https://user-images.githubusercontent.com/23367640/193998456-2aecb643-73d1-4f86-97f3-cf6bdbcf58f8.png)
![image](https://user-images.githubusercontent.com/23367640/193999141-382fd774-51bb-462e-ae2e-e1aa2813fd13.png)

Lena: the model is not trained or finetuned on that face:

![image](https://user-images.githubusercontent.com/23367640/193998666-b07a2af7-ab47-4dbe-b7da-2c44b6f2ed78.png)
![image](https://user-images.githubusercontent.com/23367640/193998792-0971d436-a0ae-4a76-ac52-6b21b376d799.png)

* **Training, no sharpening**

https://github.com/Twenkid/DeepFaceLab-SAEHDBW/blob/main/LIAE-UD-192-96-32-32-16.md


* x.9.2022 - Training DF-UD 256x256 128-32-32-16, 258 MB after 170K it.

A discovery: it happened that the model trains well even with a batch size of 4 and just 32-32 dimensions. The iteration time is comparable to training the 192x192 models. Note that the dataset is not the best regarding sharpness, especially on the Kiril's side, and for now I've been training with the 192x192 faces of Kiril, many of which extracted from 640x360 videos or from 854x480, resized from 640x360, also blurred and not sharpened/super-resolution enhanced. A future test will use sharper dataset for finetuning.

The size of the model is way below the maximum that I could fit in 750 Ti, so far: 345 MB, both 192x192 df-ud and df-udt models, so 288x288 or even 320x320? could be possible - something to try.

I don't know if the batch 4 and using just 32-32-16 dimension will work so well on lower resolution, when the features will be smaller: check it out.

~517K, 15.9.2022

![image](https://user-images.githubusercontent.com/23367640/190496593-30974d23-0381-44e7-b923-f903e92388d5.png)

~439K, 14.9.2022

![image](https://user-images.githubusercontent.com/23367640/190230048-43bf22ca-8f6a-445f-870d-919429676b0c.png)

![image](https://user-images.githubusercontent.com/23367640/190229700-60b4c4ae-5b81-4a2f-8f69-560da7dc4c41.png) ![image](https://user-images.githubusercontent.com/23367640/190229773-2241bc4f-ef92-4656-bd2c-0c793eb28207.png)
![image](https://user-images.githubusercontent.com/23367640/190229837-5c0841ce-3779-4408-9b05-4b829218fb95.png)
![image](https://user-images.githubusercontent.com/23367640/190229914-5fabf785-fd6f-4115-ab04-f3c9205dee97.png)

![0173000](https://user-images.githubusercontent.com/23367640/189473177-8600e0ca-0c4c-4bea-8e9b-67657664eec4.jpg)
Loss: [09:01:56][#172976][0745ms][0.8431][0.6401]

```
(Big values for it. time are due to he saving etc., fast ones are about 687-690 ms,
I don't push the CPU and GPU all the time and now (14.9) it goes around 716-723 ms)

[07:55:32][#295280][0701ms][0.5368][0.5851]
[08:05:08][#296115][0786ms][0.5318][0.5899]
[08:15:09][#296983][1349ms][0.5333][0.5828]
[08:25:08][#297849][0728ms][0.5277][0.5788]
[08:35:08][#298715][0765ms][0.5308][0.5818]
...
[16:55:09][#341542][0814ms][0.5063][0.5739]
[17:05:10][#342387][0830ms][0.5052][0.5729]
[17:15:09][#343227][0835ms][0.5107][0.5675]
[17:25:10][#344066][0825ms][0.5083][0.5663]
...
[21:55:10][#366704][0721ms][0.5008][0.5617]
[22:05:10][#367539][0714ms][0.4994][0.5653]
[22:15:10][#368374][0846ms][0.4997][0.5643]
...
[08:49:20][#381196][0957ms][0.4930][0.5595]
[08:58:29][#381927][0846ms][0.4921][0.5590]
[09:08:29][#382721][0929ms][0.4904][0.5592]
[09:18:29][#383520][0941ms][0.4837][0.5584]
...
[20:19:13][#435702][0743ms][0.4756][0.5525]
[20:28:45][#436491][0741ms][0.4776][0.5384]
[20:38:45][#437311][1672ms][0.4767][0.5538]
[20:48:46][#438128][1320ms][0.4766][0.5504]
[20:58:46][#438941][0727ms][0.4736][0.5382]
[21:08:45][#439747][0952ms][0.4713][0.5435]
...
[22:32:03][#515431][1127ms][0.4553][0.5283]
[22:41:06][#516131][0900ms][0.4537][0.5326]
[22:51:07][#516905][1147ms][0.4543][0.5371]
```

* xx.8.2022 - Colorization of Arnold with the POC method with Pix2Pix (Image to image) translation

The first attempts weren't good. For Stoltenberg it was reasonable that it would work, because it's in similar conditions, the Arnold dataset is extremely diverse and apparently colorization models should be trained per sequence, which is too much.

I had ideas for attempting another solution which would not use NN, but instead would directly map color faces to the grayscale ones, taking into account the shape and the facial landmarks, but lately I lack time to start implementing it.

* xx.8.2022

* Training a DF-UD 192x192 128-48-48-16 model, Kiril to Arnold (pretrained earlier), now after 36K it. without flip dst and some changes in the dataset. I expected (hoped) the new model to train in less iterations than the previous that was DF-UDT 192x19, 128-48-32-16, because of the non-symmetric number of dimensions of the encoder and decoder, but for now it seems similar, and the dataset of the Kiril ("dst") is not exactly the same.

Note, 10.9.2022: The expectations were correct. After just about 200K-210K iterations it looked similarly (LRD and sot-m for a few K in the end), the loss was a bit higher, a few humdredth. Visually there was a sequences which was significantly better - the one in front of the "parliament" background, which had the area covering the nose blinking, now it was stable.



* 26.8.2022: I'm considering a more pronouncable or/and "unique" name/alias (names/aliases) of the project. For now:
1. Arnold-DFL or Arnaud-DFL or
1. **Arnoldator** or **Arnaudator** [ArnOdator - "Арнодейтъ"] or
1.  **Arnaudatar** or **Arnoldatar** (the same pronounciation)
1. All of the above
1. **Arnoldify, Arnoldifier, ArnoldDF, ArnolDF? [Arnol-D-F]**

* ~ 10.8.2022: Experimental feature: POC version of the colorization of the output from the grayscale models during merging with additionally trained dedicated pix2pix GAN: complete prototype and merging on 10.8.2022.

![image](https://user-images.githubusercontent.com/23367640/184554576-1c308792-bf3d-497a-8061-7de10a9ae5a2.png)

* 19.8-20.8.2022: After investigation of the properties of the colorized faces, debugging of the merging, there was a successful application of an idea for stabilization of the colorized output and merging with precomputed faces (for other usages as well, e.g. prerendered 3D-models or synchronously performing faces etc.). In the video example below the output is also sharpened after merging (whole frame) - it needs to be per face only etc. or to have some antialiasing eventually.

See a merged and sharpened segment with Jens, whole frame: http://twenkid.com/v/32-stolten-color-20-8-20220.645217694677918.mp4

**Only aligned faces:**

The raw colorized face with pix2pix model without color stabilization was flickering; it was very bad, but still noticeable, especially in some moments.
https://user-images.githubusercontent.com/23367640/185765054-c012ba01-8600-4b78-9a45-3f01270237e4.mp4

After color-gamma stabilization, that artifact was gone (only the aligned face, 146 KB):
https://user-images.githubusercontent.com/23367640/185765072-bc8be151-3e7f-4758-8f5d-5d4a8f8255f9.mp4

The color-gamma stabilization is done by first probe-rendering all faces, computing their total pixel weight per frame and the average of all frames, then adjusting the gamma for each frame according to the average in order to flatten the fluctuations: if the face is too dark - it gets lighter and vice versa (_Indeed, this phenomenon itself is to show some intrinsic properties of the pix2pix model._). Finally there is sharpening and then merging is performed using the gamma-corrected faces. 
 


More info, results and code - later.

* Future work: Integration with Wav2Lip and Wav2Lip-HQ for automated lip-sync and repair of the output from the lip-sync libraries. 

* Future work: Possibly integration with RealTimeVoiceCloning? etc./other TTS engines etc.

<a name="#premiere">
 
22.6.2022

**Premiere!  Part I**

![image](https://user-images.githubusercontent.com/23367640/175982252-3d79a921-6261-4fc3-ad56-8737b812b955.png)

https://artificial-mind.blogspot.com/2022/06/arnold-schwarzenegger-governor-of.html

https://www.youtube.com/watch?v=n2JMGqxOYfA&feature=youtu.be

* **Part II**: **"Arnold meets his illegal daughter Lena Borislavova Scwarzenegger"** https://youtu.be/k1MZ8eIUjXE

![image](https://user-images.githubusercontent.com/23367640/178838680-44e7398c-a0b4-4548-9c82-9a2d3430886c.png)


* **Part III**: Combined with Wav2Lip - the mouth is synchronized; no finetuning; another pass of DFL to repair the artifacts (however there is some loss of contrast and sharpness)  https://youtu.be/4F7PB7wBEXk

![image](https://user-images.githubusercontent.com/23367640/179422086-2af2a887-4681-4f58-a68f-d16f4f6a04df.png)



* **Part IV**: Lip-syncing with finetuning repair and synthesized cloned voices (RealTimeVoiceCloning) over original script ("full directing") with Arnold, Lena Schwarzenegger and Stoltenberg    https://youtu.be/X56QkNzkkVM

![title-4-color-NATO-Arnold-reacts-to-Stoltenberg](https://user-images.githubusercontent.com/23367640/183316946-3787e880-a647-423d-b6f8-2078a514642e.jpg)



![image](https://user-images.githubusercontent.com/23367640/175983477-be704259-ae9b-41ec-a245-7ea30af6c516.png)

![image](https://user-images.githubusercontent.com/23367640/175983001-12bd3314-6119-4923-a7f3-e63707bb7427.png)



![image](https://user-images.githubusercontent.com/23367640/175983775-d8f1ae00-16d0-497f-817f-68ed7dc51204.png)

![image](https://user-images.githubusercontent.com/23367640/175986539-bec8c060-828d-41d1-8b5f-a81695213b89.png)
![image](https://user-images.githubusercontent.com/23367640/175986660-bdf0e5d9-59e7-4e02-a94c-8a0c7f83bc19.png)



![image](https://user-images.githubusercontent.com/23367640/175986888-d93e7375-a72d-4495-8785-b57870e136cb.png)

![image](https://user-images.githubusercontent.com/23367640/175986915-936591e3-d475-43fe-9ed6-5bc430a327f2.png)


# Technical details

* The model was trained on a Geforce 750 Ti 2 GB on Windows 10, created with DFL-SAEHDBW, df-udt mf 192x192 128x48x32x16, mostly batch size=6. The size on disk at the start was about 345 MB. Training began with ~494K iterations pre-training, no color transfer (it wasn't adapted yet) on a customized version of the DFL faceset: I gradually removed "bad" samples with overlapped objects etc, in the end it was about 14551 items, instead of 15843*. I didn't turn off Random warp and no LRD. The pretrained model probably could improve more, but I wanted to switch to the actual faces*. 

* Then on the two-faces training, after 509K iterations, I turned on LRD (learning rate drop-out) and SOT-M color transfer for 10Ks iterations - SOT is slow, but it improved the loss with a few iterations. Some fine-tuning experiments for a few sequences which were introduced late (the BTA ones), or had too noticeable "flashes" in the face (the "parliament"-stamps wall), also the sequences from the interview which starts after the first part of the movie when the music ends with the EU star flag (an attempt to improve the borders in semi-profile views); only a few seconds were used from the latter, mostly because I noticed a better matching  sequence, rather than due to much improved quality - the latter finetuning mostly added more contrast in the teeth, darker separation regions, but it changed the position of the eyes etc. and also the borders of the face).

![image](https://user-images.githubusercontent.com/23367640/176976685-63660241-8ea2-4ee8-967f-749bb2ec72b8.png)

* Future work: In future pretrainings I may reduce the faceset more and may split it to different types of faces and remove more of the "bad" samples in order to pretrain the model faster."Bad" are e.g. beard and moustache, old faces with deep wrinkles - they all are not reconstructed well anyway at least with the dimension I've tried so far.

* Note that the model had a natural "mask" on his face, reaching about the upper end of the mf-area, and when the model renders it darker - it actually is not a mistake and bad color transfer, LOL.

## Kiril to Arnold

![image](https://user-images.githubusercontent.com/23367640/176973091-7a392058-24fa-4383-8557-459e7b428706.png)

![image](https://user-images.githubusercontent.com/23367640/176974624-d7fb89ef-4ee2-4025-9d95-2caee2f220fe.png)
![image](https://user-images.githubusercontent.com/23367640/176974687-05d48e92-3800-42ab-8b6d-200c263a0d72.png)

# History

~ 22.4.2022 -- Minor iterface changes (more keys for save, save preview periods and auto saving;later: possible forcing generation of new previews instead of keeping the same for the whole training etc.; reviewing the code

~ 25.4.2022? --> Started working on SAEHDBW - Grayscale deepfake model; research, experiments, modifications of the channel dimensions, studying the NN model.

# Goals of the project:

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
