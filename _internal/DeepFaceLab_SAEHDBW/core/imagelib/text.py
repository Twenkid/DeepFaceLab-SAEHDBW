import localization
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import traceback #27-4-2022

pil_fonts = {}
debug_get_text_image = False #False - 30-4-2022
alter_iter_font_size = 2
#if alter_iter_font_size in os.environ: alter_iter_font_size = os.envrion["alter_iter_font_size"]  #5-5-2022

def _get_pil_font (font, size):
    global pil_fonts
    try:
        font_str_id = '%s_%d' % (font, size)
        if font_str_id not in pil_fonts.keys():
            pil_fonts[font_str_id] = ImageFont.truetype(font + ".ttf", size=size, encoding="unic")
        pil_font = pil_fonts[font_str_id]
        return pil_font
    except:
        return ImageFont.load_default()

"""
def get_text_image( shape, text, color=(1,1,1), border=0.2, font=None):
    h,w,c = shape
    try:
        pil_font = _get_pil_font( localization.get_default_ttf_font_name() , h-2)

        canvas = Image.new('RGB', (w,h) , (0,0,0) )
        draw = ImageDraw.Draw(canvas)
        offset = ( 0, 0)
        draw.text(offset, text, font=pil_font, fill=tuple((np.array(color)*255).astype(np.int)) )

        result = np.asarray(canvas) / 255

        if c > 3:
            result = np.concatenate ( (result, np.ones ((h,w,c-3)) ), axis=-1 )
        elif c < 3:
            result = result[...,0:c]
        return result
    except:
        return np.zeros ( (h,w,c) )
"""
#def get_text_image( shape, text, color=(1,1,1), border=0.2, font=None):
def get_text_image( shape, text, color=(1,1,1), border=0.2, font=None, font_size_minus_h=2):    
    h,w,c = shape
    if debug_get_text_image: print(f"get_text_image, h,w,c = shape = {h}, {w}, {c} = {shape}, color={color}")
    #NO: if c==1: color = (1,1,1) #27-4-2022
    try:
        #pil_font = _get_pil_font( localization.get_default_ttf_font_name() , h-2)
        pil_font = _get_pil_font( localization.get_default_ttf_font_name() , h-font_size_minus_h)

        canvas = Image.new('RGB', (w,h) , (0,0,0) )
        draw = ImageDraw.Draw(canvas)
        offset = ( 0, 0)
        
        if c==1: draw.text(offset, text, font=pil_font, fill=255) #(color*255,color*255,color*255)
        else: draw.text(offset, text, font=pil_font, fill=tuple((np.array(color)*255).astype(np.int)))

        result = np.asarray(canvas) / 255

        if c > 3:
            result = np.concatenate ( (result, np.ones ((h,w,c-3)) ), axis=-1 )
        #elif c < 3:
        elif c < 3: #and c!=1:
            result = result[...,0:c]
        #elif c == 1: #27-4-2022
        #    result = result[0]
        if debug_get_text_image: 
            print(f"get_text_image, result.shape={result.shape}, np.sum(result)={np.sum(result)}")
        return result
    except:      
        print("Except in get_text_image?")
        traceback.print_exc()
        return np.zeros ( (h,w,c) )
        
def draw_text( image, rect, text, color=(1,1,1), border=0.2, font=None):
    h,w,c = image.shape

    l,t,r,b = rect
    l = np.clip (l, 0, w-1)
    r = np.clip (r, 0, w-1)
    t = np.clip (t, 0, h-1)
    b = np.clip (b, 0, h-1)

    image[t:b, l:r] += get_text_image (  (b-t,r-l,c) , text, color, border, font )


def draw_text_lines (image, rect, text_lines, color=(1,1,1), border=0.2, font=None):
    text_lines_len = len(text_lines)
    if text_lines_len == 0:
        return

    l,t,r,b = rect
    h = b-t
    h_per_line = h // text_lines_len

    for i in range(0, text_lines_len):
        draw_text (image, (l, i*h_per_line, r, (i+1)*h_per_line), text_lines[i], color, border, font)

def get_draw_text_lines ( image, rect, text_lines, color=(1,1,1), border=0.2, font=None):
    image = np.zeros ( image.shape, dtype=np.float )
    draw_text_lines ( image, rect, text_lines, color, border, font)
    return image
