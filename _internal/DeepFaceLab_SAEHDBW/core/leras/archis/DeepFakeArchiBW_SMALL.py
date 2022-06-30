from core.leras import nn
#C:\DFL\DeepFaceLab_DirectX12\_internal\DeepFaceLab\core\leras\archis\__init__.py
#from .DeepFakeArchiBW import *
#Don't forget to add for other architectures!
tf = nn.tf

USER_FORCE_DOWNSCALES = 5 #5 # 4 # 4
bSkipAfterUpscaleMask = True

#6-5-2022: Try to compress part of the network for grayscale. Encoder and Decoder? Maybe be removing synchronously layers in both places?

class DeepFakeArchiBW_SMALL(nn.ArchiBase): #22-4-2022
    """
    resolution

    mod     None - default
            'quick'

    opts    ''
            ''
            't'
    """
    def __init__(self, resolution, use_fp16=False, mod=None, opts=None):
        super().__init__()
        print("DeepFakeArchiBWTEST __init__")

        if opts is None:
            opts = ''


        conv_dtype = tf.float16 if use_fp16 else tf.float32
        
        if 'c' in opts:
            def act(x, alpha=0.1):
                return x*tf.cos(x)
        else:
            def act(x, alpha=0.1):
                return tf.nn.leaky_relu(x, alpha)
                
        if mod is None:
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs ):
                    print(f'Downscale.DeepFakeArchiBW __init__ {in_ch} , {out_ch}, {kernel_size}')
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    print(f'on_build.Downscale.DeepFakeArchiBW {self.in_ch} , {self.out_ch}, {self.kernel_size}')
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch, kernel_size=self.kernel_size, strides=2, padding='SAME', dtype=conv_dtype)

                def forward(self, x):
                    print(f"Downscale.forward x = {x}")
                    x = self.conv1(x)
                    print(f'#Downscale.forward x = self.conv1(x) = {x}')
                    x = act(x, 0.1)
                    print(f'#Downscale.forward x = act(x, 0.1) = {x}')
                    return x

                def get_out_ch(self):
                    print(f"Downscale get_out_ch = {self.out_ch}")
                    return self.out_ch

            class DownscaleBlock(nn.ModelBase):
                def on_build(self, in_ch, ch, n_downscales, kernel_size):
                    print(f'on_build.DownscaleBlock.DeepFakeArchiBW {in_ch} , {ch}, {n_downscales}, {kernel_size}')
                    self.downs = []

                    last_ch = in_ch #here in BW is 1
                    print(f"last_ch = {last_ch}")
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        print(f"cur_ch = {cur_ch}")
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size))
                        print(f"Downscale(last_ch, cur_ch, kernel_size=kernel_size = {last_ch}, {cur_ch}, {kernel_size}")
                        last_ch = self.downs[-1].get_out_ch()
                        print(f"DownscaleBlock(last_ch) = {last_ch}")

                def forward(self, inp):
                    print("DownscaleBlock: forward") #add x in next iteration {x}
                    x = inp #inp should be:  last_ch, cur_ch, kernel_size=  #26-4-2022
                    for down in self.downs:
                        x = down(x)
                    return x

            class Upscale(nn.ModelBase):
                def on_build(self, in_ch, out_ch, kernel_size=3):
                    print(f'on_build.Upscale.DeepFakeArchiBW {in_ch}, {out_ch},{kernel_size}')
                    self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

                def forward(self, x):
                    print(f'forward.Upscale.DeepFakeArchiBW = {x}')
                    x = self.conv1(x); print(f'x = self.conv1(x)= {x}')
                    x = act(x, 0.1); print(f'x = act(x, 0.1) = {x} (Upscale.forward'); print(f'BEFORE: nn.depth_to_space(x,2)')
                    
                    x = nn.depth_to_space(x, 2); print(f'x = act(x, 0.1) = {x}')
                    return x

            class ResidualBlock(nn.ModelBase):
                def on_build(self, ch, kernel_size=3):
                    print(f'on_build.ResidualBlock.DeepFakeArchiBW {ch}, {kernel_size}')  
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                    print(f"self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)")
                    print(f"{self.conv1} = nn.Conv2D({ch}, {ch}, {kernel_size}, (padding=SAME), dtype={conv_dtype}")

                def forward(self, inp):
                    print(f"ResidualBlock.forward: {inp}")
                    x = self.conv1(inp); print(f"x={x}, ResidualBlock.forward; x=self.conv1(inp)")
                    x = act(x, 0.2); x = self.conv1(inp); print(f"ResidualBlock.forward; {x}=act(x,0.2")
                    x = self.conv2(x); print(f"{x} = self.conv2(x)")
                    x = act(inp + x, 0.2); print(f"{x} = act(inp+x, 0.2")
                    #print(f"x={x}")
                    return x

            class Encoder(nn.ModelBase):
                def __init__(self, in_ch, e_ch, **kwargs ):
                    print(f'__init__.Encoder.DeepFakeArchiBW in_ch, e_ch, **kwargs = {in_ch} , {e_ch}, kwargs=...')
                    self.in_ch = in_ch
                    self.e_ch = e_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    print(f'on_build.Encoder.DeepFakeArchiBW {self.in_ch} , {self.e_ch}')
                    if 't' in opts:
                        self.down1 = Downscale(self.in_ch, self.e_ch, kernel_size=5)
                        self.res1 = ResidualBlock(self.e_ch)
                        self.down2 = Downscale(self.e_ch, self.e_ch*2, kernel_size=5)
                        self.down3 = Downscale(self.e_ch*2, self.e_ch*4, kernel_size=5)
                        self.down4 = Downscale(self.e_ch*4, self.e_ch*8, kernel_size=5)
                        self.down5 = Downscale(self.e_ch*8, self.e_ch*8, kernel_size=5)
                        self.res5 = ResidualBlock(self.e_ch*8)
                    else:
                        print(f'else not t in opts: ... BEFORE self.down1 = DownscaleBlock... {self.in_ch}, {self.e_ch}')
                        #self.down1 = DownscaleBlock(self.in_ch, self.e_ch, n_downscales=4 if 't' not in opts else 5, kernel_size=5)
                        #try with downscales 1
                        self.down1 = DownscaleBlock(self.in_ch, self.e_ch, n_downscales=USER_FORCE_DOWNSCALES if 't' not in opts else 5, kernel_size=5)
                        print(f'else: ... AFTER self.down1 = DownscaleBlock... {self.in_ch}, {self.e_ch} USER_FORCE_DOWNSCALES={USER_FORCE_DOWNSCALES}')

                def forward(self, x):
                    if use_fp16:
                        x = tf.cast(x, tf.float16)

                    if 't' in opts:
                        x = self.down1(x)
                        x = self.res1(x)
                        x = self.down2(x)
                        x = self.down3(x)
                        x = self.down4(x)
                        x = self.down5(x)
                        x = self.res5(x)
                    else:
                        print(f'forward.Encoder.DeepFakeArchiBW x = {x}')
                        x = self.down1(x)
                        print(f'NEW {x}=self.down1(x)')
                    x = nn.flatten(x)
                    print(f'{x} = after nn.flaten(x)')
                    if 'u' in opts:
                        x = nn.pixel_norm(x, axes=-1)

                    if use_fp16:
                        x = tf.cast(x, tf.float32)
                    return x

                def get_out_res(self, res):
                    return res // ( (2**4) if 't' not in opts else (2**5) )

                def get_out_ch(self):
                    #return self.e_ch * 8
                    return self.e_ch * 4 #4 #6-5-2022

            lowest_dense_res = resolution // (32 if 'd' in opts else 16)
            print(f"lowest_dense_res={lowest_dense_res} in DeepFakeArchiBWTEST __ini__")

            class Inter(nn.ModelBase): #R.128 ... 3456, 64, 128 ... 6-5-2022
                def __init__(self, in_ch, ae_ch, ae_out_ch, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                    print(f"Inter(nn.ModelBase):__init__ in_ch={in_ch}, ae_ch={ae_ch}, ae_out_ch={ae_out_ch}")
                    super().__init__(**kwargs)

                def on_build(self):
                    in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch
                    print(f"Inter(nn.ModelBase): on_build in_ch={in_ch}, ae_ch={ae_ch}, ae_out_ch={ae_out_ch}")

                    self.dense1 = nn.Dense( in_ch, ae_ch ); print(f"Inter.on_build: self.dense1 = nn.Dense( in_ch, ae_ch ); {self.dense1} = nn.Dense({in_ch}, {ae_ch})")
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch ); print(f"Inter.on_build: nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch ); {self.dense2} = nn.Dense({ae_ch}, {lowest_dense_res} * {lowest_dense_res} = {lowest_dense_res*lowest_dense_res}, {ae_out_ch} )")
                    #==> out channel 
                    if 't' not in opts:
                        self.upscale1 = Upscale(ae_out_ch, ae_out_ch)
                        print(f"Inter.on_build: if 't'not in opts {opts}: self.upscale1 = Upscale(ae_out_ch, ae_out_ch) = Upscale({ae_out_ch}, {ae_out_ch})")

                def forward(self, inp):
                    x = inp; print(f"Inter:forward:x={inp}")
                    print(f"x.shape={x.shape}")
                    y = tf.keras.backend.repeat_elements(x, 2, axis=1) #OK, that solves it for now, #6-5-2022 --> but other error
                    print(f"tf.keras.backend.repeat_elements(x, 2, axis=0)--> x.shape={y.shape}")
                    print(f"{y}")
                    #x = tf.image.resize_images(x, (1736*2, 1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)                  
                    #x = self.dense1(x); print(f"Inter:forward:x=self.dense1(x)->{x})")
                    x = self.dense1(y); print(f"Inter:forward:x=self.dense1(x)->{x})")
                    #y = tf.concat([x,x], 0)
                    #y = tf.repeat(x, repeats=2)
                    #print(f"y=tf.concat(x,x)={y}")
                    #x = self.dense1(y);
                    #print(f"tf.repeat(x, repeats=2)={tf.repeat(x, repeats=2)}")
                    
                    #x = self.dense1(tf.repeat(x, repeats=2, axis=0))
                    print(f"self.dense1(tf.repeat(x, repeats=2)={x}")
                    print(f"Inter:forward:x=self.dense1(tf.concat(x,x))->{x})")
                    x = self.dense2(x); print(f"Inter:forward:x=self.dense2(x)->{x})")
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch); print(f"Inter:reshape_4D: {x}=nn.reshape_4D(x, {lowest_dense_res}, {lowest_dense_res}, {self.ae_out_ch})")

                    if use_fp16:
                        x = tf.cast(x, tf.float16)

                    if 't' not in opts:
                        x = self.upscale1(x)

                    return x

                def get_out_res(self):
                    print("Inter.get_out_res: return {lowest_dense_res}*2 if 't'not in opts {opts}, else {lowest_dense_res}")
                    return lowest_dense_res * 2 if 't' not in opts else lowest_dense_res

                def get_out_ch(self):
                    print("Inter.get_out_ch = self.ae_out_ch {self.ae_out_ch}")
                    return self.ae_out_ch

            class Decoder(nn.ModelBase):
                def on_build(self, in_ch, d_ch, d_mask_ch):
                    print(f"IN: Decoder on_build opts={opts}, {in_ch}, {d_ch}, {d_mask_ch}")
                    M0 = 8
                    M1 = 4
                    M2 = 2
                    M3 = 1
                    #Remove one Upscale and one Residual Block
                    if 't' not in opts:
                        print(f"if 't' not in opts:")
                        
                        
                        #self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                        
                        #6-5-2022
                        self.upscale0 = Upscale(in_ch, d_ch*M0, kernel_size=3)
                        #print(f"self.upscale0={self.upscale0}")
                        
                        #self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                        #self.upscale1 = Upscale(d_ch*M1, d_ch*M2, kernel_size=3)
                        #print(f"self.upscale1={self.upscale1}")
                        
                        self.upscale1 = Upscale(d_ch*M0, d_ch*M1, kernel_size=3) #, *M1
                        self.upscale2 = Upscale(d_ch*M1, d_ch*M2, kernel_size=3)
                        #print(f"self.upscale2={self.upscale2}")
                        #self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res0 = ResidualBlock(d_ch*M0, kernel_size=3)
                        #print(f"self.res0={self.res0}")
                        
                        #self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                        #print(f"self.res1={self.res1}")
                        #self.res1 = ResidualBlock(d_ch*2, kernel_size=3)
                        self.res1 = ResidualBlock(d_ch*M1, kernel_size=3)
                        self.res2 = ResidualBlock(d_ch*M2, kernel_size=3)
                        #print(f"self.res2={self.res2}")

                        self.upscalem0 = Upscale(in_ch, d_mask_ch*M0, kernel_size=3) #1) #=3)
                        print(f"self.upscalem0={self.upscalem0}")
                                                
                        #if not bSkipAfterUpscaleMask0:
                        #### Option to exlude these -- in decoder as well! TO Do #26-4-2022
                        #self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3) #1) #=3)
                        #print(f"self.upscalem1={self.upscalem0}")
                        #self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3) #1) #=3)
                        self.upscalem1 = Upscale(d_mask_ch*M0, d_mask_ch*M1, kernel_size=3) #1) #=3)
                        
                        #print(f"self.upscalem2={self.upscalem1}")
                        
                        self.upscalem2 = Upscale(d_mask_ch*M1, d_mask_ch*M2, kernel_size=3) #1) #=3)
                        

                        #t not in opts!!!! --> that had to be changed to 1 as well?
                        #self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype) 
                        #self.out_conv  = nn.Conv2D( d_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        
                        #Not in_ch, 1! #23-4-2022 #24-4 2:46
                        #self.out_conv  = nn.Conv2D( d_ch*2, in_ch, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        #self.out_conv  = nn.Conv2D( d_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        #self.out_conv  = nn.Conv2D( d_ch*2, in_ch, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        
                        
                        #self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        #self.out_conv  = nn.Conv2D( d_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        
                        
                        #self.out_conv  = nn.Conv2D( d_ch*M2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        self.out_conv  = nn.Conv2D( d_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        print(f"self.out_conv=...") #Don't print {    #print(f"self.out_conv}.")}
                        # ********************  in_ch
                        """
                        if 'd' in opts:
                            self.out_conv1 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv2 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv3 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        """
                        if 'd' in opts:
                            print(f'if d in opts...')
                            self.out_conv1 = nn.Conv2D( d_ch*2, in_ch, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            print(f'self.out_conv1 = d_ch*2 = \n {self.out_conv1}\n{d_ch}*2 = {d_ch*2}, in_ch={in_ch}, kernel_size=3')
                            self.out_conv2 = nn.Conv2D( d_ch*2, in_ch, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            print(f'self.out_conv2 = d_ch*2 =  \n {self.out_conv2}\n {d_ch}*2 = {d_ch*2}, in_ch={in_ch}, kernel_size=3')
                            self.out_conv3 = nn.Conv2D( d_ch*2, in_ch, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            print(f'self.out_conv3 = d_ch*2 =  \n {self.out_conv3}\n {d_ch}*2 = {d_ch*2}, in_ch={in_ch}, kernel_size=3')
                            
                            #self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            #self.upscalem2 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            #print(f'#DECODER on_build: self.upscalem3 = d_ch*2 =  \n {self.upscalem3}\n {d_mask_ch}, kernel_size=3')
                            
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                            #print(f'self.out_convm = d_ch*2 =  \n {self.out_convm}\n {d_mask_ch}, kernel_size=1')
                        else:
                            print("if not d in opts:")                            
                            self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                            print(f'self.out_convm = d_ch*2 =  \n {self.out_convm}\n {d_mask_ch}, kernel_size=1')
                    else:
                        self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                        print(f'self.upscale0 = {self.upscale0}')
                        #"""
                        self.upscale1 = Upscale(d_ch*8, d_ch*8, kernel_size=3)
                        print(f'self.upscale1 = {self.upscale1}')
                        self.upscale2 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                        print(f'self.upscale2 = {self.upscale2}')
                        self.upscale3 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
                        print(f'self.upscale3 = {self.upscale3}')
                       # """
                        self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                        print(f'self.res0 = {self.res0}')
                        self.res1 = ResidualBlock(d_ch*8, kernel_size=3)
                        print(f'self.res1 = {self.res1}')
                        self.res2 = ResidualBlock(d_ch*4, kernel_size=3)
                        print(f'self.res2 = {self.res2}')
                        self.res3 = ResidualBlock(d_ch*2, kernel_size=3)
                        print(f'self.res3 = {self.res3}')

                        #Maybe don't upscale because one channel only?
                        self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                        print(f'AFTER: self.upscalem0 ... {self.upscalem0}')
                        self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*8, kernel_size=3)
                        print(f'AFTER: self.upscalem1 ... {self.upscalem1}')
                        self.upscalem2 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                        print(f'AFTER: self.upscalem2 ... {self.upscalem2}')
                        self.upscalem3 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                        print(f'AFTER: self.upscalem3 ... {self.upscalem3}')
                        #self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        #NOT in_ch!!! #23-4-2022 #24-4, 1:29
                        
                        #self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        self.out_conv  = nn.Conv2D( d_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        #print(f'AFTER: self.out_conv = ... {self.out_conv}') #NOT CALLED?

                        if 'd' in opts:
                            print("DeepFakeArchiBW.py if 'd' in opts:")
                            """
                            self.out_conv1 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv2 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv3 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                            """
                            self.out_conv1 = nn.Conv2D( d_ch*2, in_ch, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            #print(f'self.out_conv1 = {self.out_conv1}')
                            self.out_conv2 = nn.Conv2D( d_ch*2, in_ch, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            #print(f'self.out_conv2 = {self.out_conv2}')
                            self.out_conv3 = nn.Conv2D( d_ch*2, in_ch, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            #print(f'self.out_conv3 = {self.out_conv3}')
                            self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            #print(f'self.upscalem4 = {self.upscalem4}')
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                            #print(f'self.out_convm = {self.out_convm}')
                            #comment self.out_convm
                            
                            ########## TWO MORE LESS? CONVS? #23-4-2022    
                            """
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                            print(f'self.out_convm ONE MORE = {self.out_convm}')
                            
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                            print(f'self.out_convm TWO MORE = {self.out_convm}')
                            """
                        else:
                            self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                            #print(f'else if not d in opts: ... self.out_convm = {self.out_convm}')
                            print(f'else if not d in opts:...')

                
                    
                def forward(self, z):
                    
                    x = self.upscale0(z); print(f"x = self.upscale0(z)\nx={x}\nz={z}")
                    x = self.res0(x); print(f"x = res0(x)\nx={x}")
                    x = self.upscale1(x); print(f"x = self.upscale1(x)\nx={x}")
                    x = self.res1(x);  print(f"x = res1(x)\nx={x}")
                    x = self.upscale2(x); print(f"x = self.upscale2(x)\nx={x}")
                    x = self.res2(x); print(f"x = res2(x)\nx={x}")

                    if 't' in opts:
                        x = self.upscale3(x)
                        x = self.res3(x)

                    if 'd' in opts:
                        #forced_nn_conv2d_ch_axis = nn.conv2d_ch_axis #
                        print(f"Decoder.forward:x=tf.nn.sigmoid(nn.depth_to_space(tf.concat((self.out_conv1(x), self.out_conv2(x), self.out_conv3(x), nn.conv2d_ch_axis), 2))")
                        print(f"x={x}")
                        """ Produces side effects? #23-4-2022 session, 24-4, 6:30
                        print(f"self.out_conv(x)={self.out_conv(x)}" )
                        print(f"self.out_conv1(x)={self.out_conv1(x)}" )
                        print(f"self.out_conv1(x)={self.out_conv2(x)}" )
                        print(f"self.out_conv1(x)={self.out_conv3(x)}" )
                        print(f"nn.conv2d_ch_axis={nn.conv2d_ch_axis}" )
                        """
                        #x = tf.nn.sigmoid( nn.depth_to_space(tf.concat( (self.out_conv(x),
                        #                                                 self.out_conv1(x),
                        #                                                 self.out_conv2(x),
                        #                                                 self.out_conv3(x)), nn.conv2d_ch_axis), 2)) # forced_nn_conv2d_ch_axis), 2) ) #nn.conv2d_ch_axis), 2) )
                        print("BEFORE  x = tf.nn.sigmoid( nn.depth_to_space(tf.concat( (self.out_conv(x),...")
                        print(f"{self.out_conv(x)}\n{self.out_conv1(x)}\n{self.out_conv2(x)}\n{self.out_conv3(x)}")
                        x = tf.nn.sigmoid( nn.depth_to_space(tf.concat( (self.out_conv(x),
                                                                         self.out_conv1(x),
                                                                         self.out_conv2(x),
                                                                         self.out_conv3(x)), nn.conv2d_ch_axis), 2)) # forced_nn_conv2d_ch_axis), 2) ) #nn.conv2d_ch_axis), 2) )
                                                                         
                        print(f"x={x}")
                    else:
                        print(f"Decoder.forward else d not in opts: x = tf.nn.sigmoid(self.out_conv(x)), x = {x}")
                        x = tf.nn.sigmoid(self.out_conv(x))


                    m = self.upscalem0(z); print(f"m=DECODER.forward.self.upscalem1(z)={m}\nz={z}")                                       
                    #if not bSkipAfterUpscaleMask0: #26-4-2022
                    m = self.upscalem1(m); print(f"m=DECODER.forward.self.upscalem1(m)={m}")
                    m = self.upscalem2(m); print(f"m=DECODER.forward.self.upscalem2(m)={m}")

                    if 't' in opts:
                        m = self.upscalem3(m)
                        if 'd' in opts:
                            m = self.upscalem4(m)
                    else:
                        if 'd' in opts:
                            #m = self.upscalem2(m)
                            m = self.upscalem3(m)
                            print(f"SMALL_ if 'd' in opts: m=self.upscalem2(m ={m}")

                    m = tf.nn.sigmoid(self.out_convm(m))
                    print(f"m=self.out_convm(m) = {m}")
                    print(f"m = tf.nn.sigmoid(self.out_convm(m))={m}")

                    if use_fp16:
                        x = tf.cast(x, tf.float32)
                        m = tf.cast(m, tf.float32)

                    return x, m

        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder
        
        print("DeepFakeArchiBWTEST: SUMMARY OF THE ARCHITECTURES")
        print("ENCODER::")
        """Not here - after creation? in Model.py?
        self.Encoder.summary() #26-4-2022
        print("INTER::")
        self.Inter.summary()
        print("DECODER::")
        self.Decoder.summary()               
        """
#nn.DeepFakeArchiBWTEST = DeepFakeArchiBWTEST #26-4-2022
nn.DeepFakeArchiBW_SMALL = DeepFakeArchiBW_SMALL #26-4-2022
print("nn.DeepFakeArchiBWTEST = DeepFakeArchiBWTEST") #26-4-2022