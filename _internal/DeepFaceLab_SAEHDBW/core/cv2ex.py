import cv2
import numpy as np
from pathlib import Path
from core.interact import interact as io
from core import imagelib 
import traceback
import os
import sys

take_a_channel = "user_bw_get_color_channel" in os.environ

if take_a_channel: user_bw_get_color_channel = int(os.environ["user_bw_get_color_channel"])
else: user_bw_get_color_channel = 2

print(f"cv2ex.take_a_channel={take_a_channel}, user_bw_get_color_channel={user_bw_get_color_channel}")

#option -- BW source

use_bw_input = "use_bw_input" in os.environ
print(f"cv2ex.bw_input = {use_bw_input}")

extract_to_bw = "extract_to_bw" in os.environ #29-4-2022 --> for extraction

#BGR

dont_save_normal = False


merge_bw_special_imwrite = "merge_bw_special_imwrite" in os.environ #18-5-2022
if merge_bw_special_imwrite: dont_save_normal = True
print ("merge_bw_special_imwrite=",merge_bw_special_imwrite)
#merge_bw_special_imwrite = True
#print ("merge_bw_special_imwrite=",merge_bw_special_imwrite)
#dont_save_normal = True

debug_cv_imwrite_bw = False #22-5-2022 don't print and show

def cv2_imread(filename, flags=cv2.IMREAD_UNCHANGED, loader_func=None, verbose=True):
    """
    allows to open non-english characters path
    """
    try:
        if loader_func is not None:
            bytes = bytearray(loader_func(filename))
        else:
            with open(filename, "rb") as stream:
                bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)        
        
        if use_bw_input: #do not convert, BW source
           #print(f"cv2_imread_color_as_grayscale.bw_input, flags={flags}")
           return cv2.imdecode(numpyarray, flags)          
        #return cv2.imdecode(numpyarray, flags)
        if extract_to_bw: return cv2.cvtColor(cv2.imdecode(numpyarray, flags), cv2.COLOR_BGR2GRAY)
        else: return cv2.imdecode(numpyarray, flags)   #will s3f work though? let's see YES
    except:
        if verbose:
            io.log_err(f"Exception occured in cv2_imread : {traceback.format_exc()}")
        return None

#dontsave = True # False #True

def cv2_imwrite(filename, img, *args): #It shouldn't matter if it's one channel grayscale? #29-4-2022
    if merge_bw_special_imwrite:
        img = img.copy(order='C') #18-5-2022
        cv2.imwrite(str(filename)+".jpg", img)
        if debug_cv_imwrite_bw:
          print(f"cv2_imwrite, img:\nimg={img},args={args}, suffix={Path(filename).suffix}")
          print(f"{filename}\n{img.shape}")
          #print(f"args={args}")
          """
          f = open(str(filename)+".bin", "wb") #Alternative, check data #18-5-2022
          f.write(img)
          f.close()
          """
        
          #img = cv2.merge((img,img,img)) #TRY #18-5-2022
          #cv2.imshow("cv2_imwrite?",img)
          #cv2.waitKey(200)
          cv2.destroyWindow("cv2_imwrite?")        
    
    if not dont_save_normal:
        if extract_to_bw: 
          print(f"extract_to_bw:")
          if len(img.shape<3): img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, buf = cv2.imencode( Path(filename).suffix, img, *args)
        if ret == True:
            try:
                with open(filename, "wb") as stream:
                    stream.write( buf )
            except:
                print(sys.exc_info())
                pass

def cv2_resize(x, *args, **kwargs):
    h,w,c = x.shape
    x = cv2.resize(x, *args, **kwargs)
    
    x = imagelib.normalize_channels(x, c)
    return x
    
#22-4-2022, Twenkid
def cv2_imread_color_as_grayscale(filename, flags=cv2.IMREAD_UNCHANGED, loader_func=None, verbose=True):  
    #return cv2_imread(filename, flags, loader_func,verbose) #26-4-2022
    """
    allows to open non-english characters path
    """
    try:
        if loader_func is not None:
            bytes = bytearray(loader_func(filename))
        else:
            with open(filename, "rb") as stream:
                bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        if use_bw_input: #do not convert, BW source
          #print(f"cv2_imread_color_as_grayscale.bw_input, flags={flags}")
          return cv2.imdecode(numpyarray, flags)
        if take_a_channel:
          #print(f"cv2_imread_color_as_grayscale.take_a_channel, flags={flags}")
          return cv2.imdecode(numpyarray, flags)[:,:, user_bw_get_color_channel] #user_bw_get_color_channel,:] 
        else:             
            return cv2.cvtColor(cv2.imdecode(numpyarray, flags), cv2.COLOR_BGR2GRAY) #slowest
        #More efficient: just one channel or convert/extract to BW directly #23-4-2022
    except:
        if verbose:
            io.log_err(f"Exception occured in cv2_imread : {traceback.format_exc()}")
        return None
        