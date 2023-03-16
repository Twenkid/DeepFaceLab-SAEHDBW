# Various notes, diary, log | Записки, дневник на Arnoldifier - Deepfacelab-SAEHDBW by Twenkid

* 29.4.2022 7:50
На ДФЛ-то новия ми черно-бял модел още се ускорява - в първата версия четеше цветни образи и ги преобразуваше в черно-бели, но това забавя. Сега направих да се настройва да може да приготвя (операцията EXTRACT) и после да чете направо чернобели едноканални jpg, иначе ги преобразува всеки път, това се прави от процесора на всяка итерация/бач.
Доколкото гледах и всеки път има и преоразмеряване, защото може да е променен режима mf, f, wf, ... и не знае дали не са сменени снимките. 
За големи снимки се губи много време за тези процеси,  напр. в pretrain  набора от инсталацията са 768х768, това явно мъчи поне моя процесор и натоварваше някъде на 50% или по-малко, с периодични "дупки" както бях пращал от HWInfo графика.
То и за цветно няма смисъл толкова огромни снимки, може първо да се смалят до по-съответен на целевото обучение размер.
Като са направени оригинално черно-бели и на по-разумен размер 384х384 се ускори два пъти и уплътни натоварването.

https://github.com/chervonij/DFL-Colab/releases/download/pretrain_CelebA.zip
https://github.com/chervonij/DFL-Colab/releases/download/pretrain_GenericFFHQ/pretrain_FFHQ.zip

pretrain_link = pretrain_link+"pretrain_GenericFFHQ/pretrain_FFHQ.zip" if Download_FFHQ else pretrain_link+"pretrain-CelebA/pretrain_CelebA.zip"

24.4.2022 23:16 ... old SAEHD test, res 96, 96,64,64, 22

C:\DFL\DeepFaceLab_DirectX12\_internal\DeepFaceLab\core\imagelib\text.py
debug_get_text_image = False

Returns lines for the preview window.
Modified to write and return single-channel images.

```
Setting the CUDA memory higher/up to almost the maximum (1.98 GB) didn't crash the system, it was just OOM errors. However I didn't manage to fit a bigger batch size - I'm not sure is this amount just "info" and Windows reserving what it wants anyway.

Initially there was another memory issue with running big models, one which I run with a batch size 6 on DirectX12, but managed to fit only a batch=4 in CUDA, I assumed CUDA was taking more memory (or it's really not cleared properly).

However after performing the procedure which I mentioned in the previous message:
1) PC-->Sleep
2) Connect the monitor to the integrated GPU output
3) Resume 
4) GPU memory usage = 0%

It encompassed the batch 6, but can't fit 7 even with 1.98 GB set manually, and batch 6 fits within the default Windows 1.45 GB so it seems this size is just a "suggestion"?

Running a big model seems to be about 50% faster, cool! (A refactored DFL  with a custom model/mode which I added recenyly, but I'll share that in another topic after experimenting some more.)

...
-- a modified version of DFL which I recently developed in order to incorporate grayscale models, for now it's only SAEHDBW, which is the same like SAEHD, but accepting 1-channel input. I wished to create compressed SAEHD, to reduce some of the layers, but on the first attempts I didn't manage to adjust the architecture correctly.
-- I'll share it after playing a bit more, if anyone could try this too.
-- It would be interesting to see how big resolutions and detail in grayscale could be fit.

There was a lot of refactoring in the other parts of the code in order to encompass the grayscale images, so far with environment variables etc., because it was both simpler and easier to do and didn't require to add trailing changes within the interactions between the components of the system.

Most of the work was studying the code and refactoring all places of the system to deal with grayscale images and to properly display the preview etc.

```

* C:\DFL\DeepFaceLab_DirectX12\SAEHDBW_doc\Commands.txt
Z:\DFL-SAEHDBW-Commands.txt

SAEHDBW

Extract:


Resize:

Resize to grayscale and optionally change the face type: interactively choose new resolution and face type:

@echo off
call _internal\setenv.bat
SET resize_color_to_grayscale=1

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" facesettool resize ^
    --input-dir "Z:\s"

pause

Only convert to grayscale, keep the same resolution and face type: no resize and no change of the face type (the interactively input values are ignored):

@echo off
call _internal\setenv.bat
SET resize_color_to_grayscale=1
SET only_convert_to_grayscale=1

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" facesettool resize ^
    --input-dir "Z:\s"

pause


Use grayscale input and resize and optionally change the face type:

@echo off
call _internal\setenv.bat
SET use_bw_input=1

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" facesettool resize ^
    --input-dir "Z:\s"

pause

* #Заби неочаквано CUDA недостатъчно памет pix2pix което доведе и до рестарт и загубих много несъхранени неща - за Arnold/NATO, и др., pix2pix  пр. ТЪПО!

(...) ADD Continue ...

## Colorization + Pix2Pix Model, 13-8-2022

? Или просто копира 
от цветния файл? мн. стабилен цветът?

№ Вече стана, но един кадър да се измести напред цветното - от 4272, а не 4271

№Трябва обаче да дообуча върху кадрите от филма -- има извлечени около 540 засега, resize ...  (трябва повече)

№Всъщност в 17 беше използвало цветните кадри като управляващо лице.
