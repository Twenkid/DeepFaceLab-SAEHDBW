import numpy as np
import cv2
from PIL import Image

#use_bw_input = "use_bw_input" in os.environ  #23-5-2022

#n_colors = [0..256]
def reduce_colors (img_bgr, n_colors):
    img_rgb = (img_bgr[...,::-1] * 255.0).astype(np.uint8)
    img_rgb_pil = Image.fromarray(img_rgb)
    img_rgb_pil_p = img_rgb_pil.convert('P', palette=Image.ADAPTIVE, colors=n_colors)

    img_rgb_p = img_rgb_pil_p.convert('RGB')
    img_bgr = cv2.cvtColor( np.array(img_rgb_p, dtype=np.float32) / 255.0, cv2.COLOR_RGB2BGR )

    return img_bgr

def reduce_colors_bw (img, n_colors): #23-5-2022
    img = (img*255.0).astype(np.uint8)
    print(f"reduce_colors_bw={img.shape}")
    img_rgb_pil = Image.fromarray(img[0]) #shape is 3D h,w,1, should be 2D
    img_rgb_pil_p = img_rgb_pil.convert('P', palette=Image.ADAPTIVE, colors=n_colors)
    img = np.array(img_rgb_pil_p, dtype=np.float32)/255.0
    #img_rgb_p = img_rgb_pil_p.convert('RGB')
    #img = cv2.cvtColor( np.array(img_rgb_p, dtype=np.float32) / 255.0, cv2.COLOR_RGB2BGR )

    return img
