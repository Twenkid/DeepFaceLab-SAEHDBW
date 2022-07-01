DeepFaceLab-SAEHDBW

https://github.com/Twenkid/DeepFaceLab-SAEHDBW

Contact, support, collaboration:

https://artificial-mind.blogspot.com
http://twenkid.com

Notes version: 1.7.2022

ABOUT

DeepFaceLab-SAEHDBW is an extension of iperov's DeepFaceLab 2.0, which adds the grayscale architecture SAEHDBW for theoretically 3 times higher performance for black and white images.

This extension is created by Todor Arnaudov - "Twenkid".

See a demo film: "Arnold Schwarzenegger: The Governor of Bulgaria"

https://artificial-mind.blogspot.com/2022/06/arnold-schwarzenegger-governor-of.html

https://www.youtube.com/watch?v=n2JMGqxOYfA

Note that the model for the movie is trained on just a Geforce 750 Ti on Windows 10. It uses DF-UDT 192-128-48-32-16. Windows  allows allocation of only up to 1.45 GB out of 2 GB and this model is smaller even than that. Usual batch size: 6 in GPU.

I pretrained also DF-UD 192-128-48-48-16 - expecting possibly higher quality results - and a higher resolution DF-UDT-256-96-32-32-16 etc. ... See images in the repository. These models are not yet tested with target faces though.

INSTALLATION

1) Install DFL 2.0 build (either CUDA or DIRECTX12 version): 

https://github.com/iperov/DeepFaceLab

https://disk.yandex.ru/d/7i5XTKIKVg5UUg/DeepFaceLab

2) Copy the content of the folder (C:\DFL\... is where you have installed the library)

C:\DFL\DeepFaceLab_NVIDIA_up_to_RTX2080Ti\_internal\DeepFaceLab

to a new folder:

C:\DFL\DeepFaceLab_NVIDIA_up_to_RTX2080Ti\_internal\DeepFaceLab_SAEHDBW\

3) Save SAEHDBW source files in that new directory:

C:\DFL\DeepFaceLab_NVIDIA_up_to_RTX2080Ti\_internal\DeepFaceLab_SAEHDBW\

Confirm the overwrite of of the modified files.

4) Copy the sample .bat files to the root folder of DFL2:

C:\DFL\DeepFaceLab_NVIDIA_up_to_RTX2080Ti\


Before invoking them, edit the content with the proper paths!


...

Note that the original code from the installation is in:

C:\DFL\DeepFaceLab_NVIDIA_up_to_RTX2080Ti\_internal\DeepFaceLab\

The SAEHDBW is in a parallel directory in order to keep both for reference and in case there are problems with SAEHDBW.

You may copy also the setenv.bat which has a line added for the path to the SAEHDBW version:
SET DFL_ROOT_SAEHDBW=%INTERNAL%\DeepFaceLab_SAEHDBW

And then use that varibale in the scripts (see examples)

This line can be added manually to the individual batch scripts also, or you may change the main variable:

SET DFL_ROOT=%INTERNAL%\DeepFaceLab
to
SET DFL_ROOT=%INTERNAL%\DeepFaceLab_SAEHDBW

See the sample batch files in order to use the SAEHDBW architecture.

...

The DFL library's infrastructure is adapted for black-and-white training, which includes for example the preview window, merging (only automatic, no interactive mode yet), color transfers (only SOT and LCT), resizing with conversion to grayscale - it needs to use the library functions, because the aligned files contain metadata and a batch resizing with a third-party program such as IrfanView etc. removes the metadata.

INTERFACE ADDITIONS

A few changes in the interface for more convenience:

When entering options, the numbers [1/0] can also be used for [y/n]

In the preview window:

1,l: changes the ranges of the loss graph and more steps are added 
2,s: save
[3-9], p: update preview (3,4,5,...9 - in order to make it easier to hit a button)

Info about the currently trained model is printed in a line in the preview window.

A lot of debug info is printed in the console during the model's initialization, merging etc., there are flags as env.variables - this info has to be made more pretty, cleaned etc.; the current state was the result of quick experimentation and study of the code.

OPTIONS SET AS ENVIRONMENT VARIABLES

Many options are set as environment variables, there is some inter-module communication, also implemented with data stored in an environment variable. It is probably not the most pretty way for some developers and there is repetition of code in some cases (the heading of the python scripts checking for the environment variables), however it was simple to implement and it didn't introduce the need to adjust the argument parser and to have trailing changes between classes and calls, with which I was not familiar yet. I wanted to make it work as quickly as possible.

As mentioned above, training could be either with color images, which are converted on the fly in each iteration, or with grayscale images: the latter is recommended, especially if you use high resolution faces; the aligned black-and-white faces are created with the proper resize script, see sample files.

LIMITATIONS:

SAEHDBW model - only this SAEHD one-channel model is available for grayscale, no AMP, QUICK96, RTM, XSeg support.

COLOR TRANSFERS:

Only SOT-M and LCT. From my test I recommend SOT-M. However note that it is slow and perhaps it is not a good idea to apply it during the entire training process. In my training experience I first pretrained and trained the model (say ~ 450K - 500K iterations/batch 6 for the sample video), then about the same number for the target faces, and after that I run the SOT color transfer mode for some 10K-20K-etc. iterations or for more - it improved quickly.

"Color" in grayscale is brightness/contrast match with the original face. 

MERGING

WARNING: The interactive mode for merging is not adapted for grayscale yet! For now use only the automatic mode: overlay, dst, ... 


RESIZING

For higher performance, the color extracted input can be resized to grayscale with a script with proper env.vars - see the examples.

CUDA ISSUE WORK AROUND

device_lib.list_local_devices() doesn't return in the CUDA build up to 2080

https://github.com/iperov/DeepFaceLab/issues/5515

One DFL/CUDA possible issue which is not DFL itself fault, but may prevent you from using the CUDA build with the color models as well:

If using the CUDA version, a function that checks for CUDA devices didn't return on my machine for un unknown reason. If you have that issue, my solution, implemented in the library, was to skip that call and to manually set some sample values for device title and memory. The default in the sample batch files is for 750 Ti, but you should not worry if you don't change it, as it seems it is only for information purposes for the program interface; setting a higher value doesn't let me working with bigger models, so if you like an older GPU, you may type in "3dfx Voodoo" and "1024" bytes if you wish. 

However the work-around is only for one CUDA card and accesses only the 0-th element, if you have many cards more items have to be added to the list.

Version:

22.4.2022 - 2x.5.2022 + a bit 6.2022

Release: 1.7.2022
