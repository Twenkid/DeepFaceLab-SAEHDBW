# https://colab.research.google.com/drive/1R0k1faojNm7oYeJhlU1GTLe_Xvf6SJPz#scrollTo=YfIk2es3hJEd
# Converted to local and edited etc. by Todor Arnaudov - Twenkid, #8-8-2022
# Colorization for DFL-SAEHDBW (DeepfaceLab grayscale model, or "Arnoldifier":
# Watch the Youtube series: Arnold Schwarzenegger: The Governor of Bulgaria)
# Watch the colorized example in the repository:
# https://github.com/Twenkid/DeepFaceLab-SAEHDBW
# This script is training a colorization pix-to-pix model: 
# A grayscale dataset to color dataset (the grayscale images are produced from the color extracted faces and are put
# both in one image. There was a tool for the preparation of the dataset, which is not in this file, to be added.
# Then Merging is done in two passes using special environment variables.
# 1. The Grayscale model generates not-warped grayscale images and stores them to disk.
# 2. Another Merging pass is called with required parameters, suggesting to bypass the face generation
#     and instead to colorize a set of prerendered faces which are read from the disk.
# This sample code works at a fixed resolution of 128x128 (half of 256x256 = 256//2 etc.) 
# which fits in 750 Ti on Windows 10. If you remove the // 2 it will turn into a 256 model.
#
# WARNING: Experimental code - sorry I don't have time to clear it and arrange it properly right now.
# In order to be used in DFL, another piece of code is needed in the Arnoldifier Library
# with the sample batch files and instructions about how to set up the command line parameters: To be Done... [Note, 23-3-2023]
#
# PIX2_128 --> CONVERT THE NETWORK
# 13-8-2022 to do: Add command line arg --> filename, not fixed names; allow to call more instances etc.; 
# Save as BMP (to RAM DISK, parameter)/check time - faster? or high quality jpg
# Path in development: C:\DFL\DeepFaceLab_DirectX12\_internal\DeepFaceLab\colorize\pix_128.pys

import tensorflow as tf

import os
import pathlib
import time
import datetime

#from matplotlib import pyplot as plt
#from IPython import display

#import os
import cv2
import sys

#SET!
checkpoint_dir = 'C:\BACKUP\23-8-2022-PIX2PIX-Colorize-NATO\T\training_checkpoints_128\\'

checkpoint_dir = "Z:\\training_checkpoints_128\\"

use_train = False #12-8-2022

print("PIX_128.py!!!!!1")
tf.compat.v1.enable_eager_execution() #10-8-2022
tfprintloss = False

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#NOT ENOUGH GPU RAM

#adjust to 192x...

dataset_name = "maps" #@param ["cityscapes", "edges2handbags", "edges2shoes", "facades", "maps", "night2day"]

dataset_name = "facades" #2

_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

bDownload = 0 #1 #False #True #False  maps -- too big
show_sample_image = False
show_before_start = False
show = True

if bDownload:
  path_to_zip = tf.keras.utils.get_file(
    fname=f"{dataset_name}.tar.gz",
    origin=_URL,
    extract=True)
  path_to_zip  = pathlib.Path(path_to_zip)
#OK
else:
  #path_to_zip = pathlib.Path("T:\\maps.tar.gz")
  path_to_zip = pathlib.Path("T:\\maps.tar\\")

#PATH = path_to_zip.parent/dataset_name
#PATH ="C:\Users\toshb\.keras\datasets\facades"

#for training
PATH = pathlib.Path("Z:\\pix128\\")
#PATH = pathlib.Path("Z:\\pix128\\"
"""
PATH = pathlib.Path("C:\\Users\\toshb\\.keras\\datasets\\facades\\")
PATH = pathlib.Path("T:\\p2p3\\")
PATH = pathlib.Path("T:\\p2p_128\\")
PATH = pathlib.Path("T:\\pix128\\")
print(PATH)
"""

#OK

#PATH = pathlib.Path("T:\\maps.tar\\maps\\")

#print(PATH)

#print("...")

#list(PATH.parent.iterdir())


#sample_image = tf.io.read_file(str(PATH / 'train/1.jpg'))
#sample_image = tf.io.read_file(str(PATH / '\\train\\1.jpg'))

"""
sample_image = tf.io.read_file(str(PATH / 'train\\1.jpg'))
sample_image = tf.io.decode_jpeg(sample_image)
print(sample_image.shape)
"""

"""
if show_sample_image:
  plt.figure()
  plt.imshow(sample_image)
  plt.show()
"""

#OK
def load_jpg(image_file, to_rgb=False):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  print(image)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  #w = w // 2
  input_image = image[:, :, :]
  #real_image = image[:, :w, :]
  
  if to_rgb: input_image = tf.image.grayscale_to_rgb(input_image)

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  #real_image = tf.cast(real_image, tf.float32)

  return input_image#, real_image
  
  
#OK
def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

#inp  
## inp, re = load(str(PATH / 'train/1.jpg'))
# Casting to int for matplotlib to display the images

if show_sample_image:
  plt.figure()
  plt.imshow(inp / 255.0)
  plt.figure()
  plt.imshow(re / 255.0)
  
  
# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size

 #TO DO: scan the images and check shape etc. for different datasets
IMG_WIDTH = 256 // 2 #192 #256  --> 128
IMG_HEIGHT = 256 //2 #192 #256

print(IMG_HEIGHT, IMG_WIDTH)
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image
  
def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]
# Normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image
@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  res_y = int(IMG_WIDTH* (1.0 + 286/256)) #286 #adjust etc. percentage
  res_x = int(IMG_WIDTH* (1.0 + 286/256)) #ERROR: 256 + 286/256 etc. !!!
  print(f"res_y, res_x = {res_y}, {res_x}")
  #input_image, real_image = resize(input_image, real_image, 286, 286)
  input_image, real_image = resize(input_image, real_image, res_y, res_x)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image
  
#plt.figure(figsize=(6, 6))

"""
if show_sample_image:
    for i in range(4): #4
      rj_inp, rj_re = random_jitter(inp, re)
      plt.subplot(2, 2, i + 1)
      plt.imshow(rj_inp / 255.0)
      plt.axis('off')
    plt.show()
"""
#OK!
#4:50

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
  
def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
  
 
if use_train: train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg')) #CHECK

if use_train: train_dataset = train_dataset.map(load_image_train)

#train_dataset = train_dataset.map(load_image_train,                                 num_parallel_calls=tf.data.AUTOTUNE)

if use_train: train_dataset = train_dataset.shuffle(BUFFER_SIZE)

if use_train: train_dataset = train_dataset.batch(BATCH_SIZE)

"""
try:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
  
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)
"""

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result
  
"""  
down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)
"""

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
"""  
up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)
"""
def Generator():
  inputs = tf.keras.layers.Input(shape=[256 // 2, 256 // 2, 3])

  down_stack = [
    downsample(64 // 2, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128 // 2, 4),  # (batch_size, 64, 64, 128)
    downsample(256 // 2, 4),  # (batch_size, 32, 32, 256)
    downsample(512 // 2, 4),  # (batch_size, 16, 16, 512)
    downsample(512 // 2, 4),  # (batch_size, 8, 8, 512)
    downsample(512 // 2, 4),  # (batch_size, 4, 4, 512)
    downsample(512 // 2, 4),  # (batch_size, 2, 2, 512)
    #downsample(512 // 2, 4),  # (batch_size, 1, 1, 512)  #one step less?***
  ]

  up_stack = [
    #upsample(512 // 2, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)  #one step less?***
    upsample(512 // 2, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512 // 2, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512 // 2, 4),  # (batch_size, 16, 16, 1024)
    upsample(256 // 2, 4),  # (batch_size, 32, 32, 512)
    upsample(128 // 2, 4),  # (batch_size, 64, 64, 256)
    upsample(64 // 2, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
  
#4:57


dpi = 80 #default 64 - too low
generator = Generator()
"""
tf.keras.utils.plot_model(generator, to_file='generator_128.png', show_shapes=True, dpi=dpi)
#os.system("model.png")
"""

"""
#usually returns to a JUpyter notebook

tf.keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False
)
"""

"""
gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
"""

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
  
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256 // 2, 256 // 2, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256 // 2, 256 // 2, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64 // 2, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128 // 2, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256 // 2, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512 // 2, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
  
  
discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, to_file="discriminator.png", show_shapes=True, dpi=64)


#tf.keras.utils.plot_model(discriminator, to_file="discriminator_128.png", show_shapes=True, dpi=dpi)

"""
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
"""

#plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
#plt.colorbar()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss
  

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
#Set in the header
#checkpoint_dir = 'T:\\training_checkpoints_128\\'
#checkpoint_dir = 'Z:\\training_checkpoints_128\\'
#checkpoint_dir = 'C:\BACKUP\23-8-2022-PIX2PIX-Colorize-NATO\T\training_checkpoints_128\\'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
                                 
#Check how many are saved                                 
                       
prev_plt = None
def generate_images(model, test_input, tar, show=True, name=None): #name - file, extension etc.
  
  prediction = model(test_input, training=True)
  #plt.figure(figsize=(15, 15))
  #plt.figure(figsize=(10, 10))
  
  
  ##plt.figure(figsize=(9, 4))
  
  #plt.ion() #? for interactive

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  """
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    if show:
      plt.imshow(display_list[i] * 0.5 + 0.5)    
      plt.axis('off')      
    if name!=None: plt.savefig(name)
    #input("Press [enter] to continue.")
  plt.pause(0.001)
  plt.show() #block = False) #TEsT
  """
  
  #cv2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) this is from CV2 to mATPLOTLIB
  #first set image from plt somehow cv2.imshow("GENERATE",cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 
  #may be not saved already! 
  #if name!=None: cv2.imshow("GENERATE",name)  #read it from the file
  
"""
for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)
log_dir="logs/"
"""

#summary_writer = tf.summary.create_file_writer(
#  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
losses_all = None #1,1,1 #9-8-2022
disc_loss_all = None
  
@tf.function
def train_step(input_image, target, step):
  global losses_all, disc_loss_all
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    losses_all = (gen_total_loss, gen_gan_loss, gen_l1_loss) #9-8
    disc_loss_all = disc_loss #9-8
    if tfprintloss:
      tf.print(gen_total_loss)
      tf.print(gen_gan_loss)
      tf.print(gen_l1_loss)
      tf.print(disc_loss)      

    

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


if show_before_start: plt.show() #for not blocking matplotlib
    
step_to_generate = 1000# 1000 #100
target_iter = 50000 #10000
step_to_checkpoint = 3000 #1000# 9999999 #5000

def fit(train_ds, test_ds, steps):
  global losses_all, disc_loss_all
  example_input, example_target = next(iter(test_ds.take(3)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % step_to_generate == 0:
      display.clear_output(wait=True)

      if step != 0:
        #print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
        print(f'Time taken for {step_to_generate} steps: {time.time()-start:.2f} sec\n')
      
      example_input, example_target = next(iter(test_ds.take(3)))
      
      start = time.time()
      
      ###plt.close() #Close current
      
      #generate_images(generator, example_input, example_target, False, f"facade_{step}.jpg") #False - don't show
      generate_images(generator, example_input, example_target, True, f"colorize_128_{step}.jpg") #False - don't show
      #img = cv2.imread(f"facade_{step}.jpg")  #read it from the file
      #if img: cv2.imshow("GENERATE",f"facade_{step}.jpg") 
      
      #print(f"Step: {step//step_to_generate}k")
      print(f"Step: {step}")
      #if losses_all != None:
     
      #print("LOSSES: losses_all = (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss_all)", str(strgen_total_loss), str(gen_gan_loss), str(gen_l1_loss), str(disc_loss_all))
      """
      if disc_loss_all!=None:
        print("LOSSES: losses_all: gen_total, gen_gan, gen_l1, disc_loss")
        gen_total_loss, gen_gan_loss, gen_l1_loss = losses_all
        tf.print(gen_total_loss)
        tf.print(gen_gan_loss)
        tf.print(gen_l1_loss)
        tf.print(disc_loss_all)      
      """
    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every ... steps #def. 5K
    if (step + 1) % step_to_checkpoint == 0:
      #checkpoint.save(file_prefix=checkpoint_prefix)  #without manager
      manager.save() #file_prefix=checkpoint_prefix)
      

"""
!tensorboard dev upload --logdir {log_dir}
display.IFrame(
    src="https://tensorboard.dev/experiment/lZ0C6FONROaUMfjYkVyJqw",
    width="100%",
    height="1000px")
!ls {checkpoint_dir}
# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# Run the trained model on a few examples from the test set
for inp, tar in test_dataset.take(5):
  generate_images(generator, inp, tar)

"""
target_iter = 999999999 #60000 #7000
#fit(train_dataset, test_dataset, steps=40000) 

restore = True
train = False #9-8-2022
gen = True #False
predict = True

if restore:
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

print("Press a Key...")
#k = cv2.waitKey(0)
#k = input("Press EN ... ")

"""
if train:
  fit(train_dataset, test_dataset, steps=target_iter)  #100
"""
#checkpoint.save(file_prefix=checkpoint_prefix)  without manager

save_end = False

if save_end: #but needs 700 MB more for 256x256
  manager.save() #file_prefix=checkpoint_prefix)

"""
if gen:
    print("GEN")
    nn=0
    for inp, tar in test_dataset.take(1):
      try:
        generate_images(generator, inp, tar, True, "T:\\generated"+str(nn)+".jpg")
        cv2.waitKey(100)
        plt.close() #Close current
      except: print("EXC"); pass      
      print(nn)
      nn=nn+1      
      if nn>100: break    
print("OK")
"""

if predict:
  print("PREDICT")
  #test_input = load_jpg("T:\\stolt_all_aligned\\stolt-2385_0.jpg", True) # cv2.imread("stolt-2395_0") #128x128  
  #test_input = load_jpg("T:\\stolt_all_aligned\stolt-2648_0.jpg", True) # cv2.imread("stolt-2395_0") #128x128  
    
  #input_image = "T:\\input_image.png"  
  #predicted_image = "T:\\predicted.png"
  
  #input_image = "T:\\input_image.bmp"  
  #predicted_image = "T:\\predicted.bmp" 
  print(f"pix_128.py:\n{sys.argv}")
  input_image = sys.argv[1]
  predicted_image = sys.argv[2]
    
  test_input = load_jpg(input_image, True)
  print(f"IN/OUT={input_image}, {predicted_image}")
  
  #test_input = load_jpg("T:\\6bw.jpg", True) # cv2.imread("stolt-2395_0") #128x128  
  print(test_input)
  height_new = 128
  width_new = 128
  test_input = tf.image.resize(test_input, [height_new, width_new],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  test_input = (test_input / 127.5) - 1 #normalize
  ####plt.imshow(test_input/255.0)

  #try:
  #new_dataset = tf.data.Dataset("T:\\stolt_all_aligned\\stolt-2385_0.jpg")
  #list_files(str(PATH / 'test/*.jpg'))
  #except tf.errors.InvalidArgumentError:
  #    test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))  
  #new_dataset = new_dataset.map(load_jpg)
  #new_dataset = new_dataset.batch(1)      
  
  """
  filenames = tf.constant(["T:\\stolt_all_aligned\\stolt-2385_0.jpg"])  
  #new_dataset = tf.data.Dataset(["T:\\stolt_all_aligned\\stolt-2385_0.jpg"])
  new_dataset = tf.data.Dataset(filenames,[test_input])
  #new_dataset = tf.data.Dataset.from_tensor_slices((filenames, load_jpg)
  new_dataset = new_dataset.map(load_jpg)
  
  #tf.data.Dataset(load_jpg("T:\\stolt_all_aligned\\stolt-2385_0.jpg"))
  test_input = next(iter(new_dataset.take(1)))
  """
  
  #tf.keras.utils.save_img("T:\\DFL-input_bababa.png", test_input)   #KERAS 2
  print(dir(tf.compat.v1.keras))
  print(dir(tf.compat.v1.keras.utils))
   
  # tf.compat.v1.keras.utils.save_img("T:\\DFL-input_bababa.png", test_input)  
  
  
  #tf.keras.preprocessing.image.save_img("T:\\DFL-input_bababa.png", test_input)
  
 
  
  #test_input /= 255.0
  #plt.imshow(test_input/255.0)
  
  print(test_input)
  #prediction = generator(test_input, training=True)
  #prediction = generator(tf.expand_dims(test_input, axis=0), training=True)  
  prediction = generator(test_input[tf.newaxis, ...], training=True) #False) #True) #False)  
  #prediction = generator(test_input, training=True) #False) #True) #False)  
  #prediction = generator(test_input[:], training=True) #False)  
  print("PREDICTION:")
  print(prediction)
  ##### plt.imshow(prediction[0, ...]* 0.5 + 0.5)
  
  
  #tf.keras.utils.save_img("T:\\DFL-predicted_bababa.png", tf.squeeze(prediction)) #prediction)
  
  #tf.keras.preprocessing.image.save_img("T:\\DFL-input_bababa.png", test_input)  
  
  
  #gen_output = generator(inp[tf.newaxis, ...], training=False)
  
  # print(tf.expand_dims(test_input, axis=0))
  
  #tf.keras.utils.save_img("T:\\predicted_bababa.png", prediction)
  # tf.keras.utils.save_img("T:\\predicted_bababa.png",  tf.squeeze(prediction))
  # print(tf.squeeze(prediction))
  # plt.figure(figsize=(7, 5))  
  # plt.title("PREDICTED COLOR")
  # display_list = [test_input[0], test_input[0], prediction[0]]
  # title = ['Input Image', "Input", 'Predicted Image']  
  """
    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      if show:
        plt.imshow(display_list[i] * 0.5 + 0.5)    
        plt.axis('off')      
      #if name!=None: plt.savefig(name)
      plt.savefig("T:\gyzzzzzzzz.jpg")
      #input("Press [enter] to continue.")
    plt.pause(0.001)
    plt.show() #block = False) #TEsT
  """
  #print(tf.expand_dims(test_input, axis=0))
    
  #tf.keras.utils.save_img("T:\\predicted_bababa.png", prediction)
  #tf.keras.utils.save_img("T:\\predicted_bababa.png", tf.squeeze(prediction))
  #tf.keras.utils.save_img("T:\\KUR-DFL-predicted_bababa.png", tf.squeeze(prediction))
  
  #tf.keras.preprocessing.image.save_img("T:\\KUR-DFL-predicted_bababa.png", tf.squeeze(prediction))
  
  tf.keras.preprocessing.image.save_img(predicted_image, tf.squeeze(prediction))

  #### k = input("Press EN ... ")                          
  
  #cv2.imshow("COLOR?",prediction)
  #cv2.waitKey(0)

####input("ENTER...")


#70K 3:23 #9-8-2022
#73K 3:28
#+59K? or 60K
#~ 133K 5:07
#7:0x - беше без надзор, забравих че е толкова бързо! по-голям брой итерации 9999999...
#+15К  - 148K
#+136K - 272 - забелязах, че имаше грешни примери, ЧБ в ляво! - поправих, ако не съм пропуснал някой
#+18K = 300K, 10:15 + 3K = 303
#+86K - още няколко грешни, гледа надолу   ok. 389K 12:05
#+123K ok. 512K 14:14


def colorize(test_input): #10-8-2022
      
    if predict:
      print("PREDICT")
      #test_input = load_jpg("T:\\stolt_all_aligned\\stolt-2385_0.jpg", True) # cv2.imread("stolt-2395_0") #128x128  
      
      
      #test_input = load_jpg("T:\\stolt_all_aligned\stolt-2648_0.jpg", True) 
      
      # cv2.imread("stolt-2395_0") #128x128  
      #test_input = load_jpg("T:\\6bw.jpg", True) # cv2.imread("stolt-2395_0") #128x128  
      print(test_input)
      height_new = 128
      width_new = 128
      
      test_input = tf.image.resize(test_input, [height_new, width_new],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      test_input = (test_input / 127.5) - 1 #normalize
      ####plt.imshow(test_input/255.0)

      #try:
      #new_dataset = tf.data.Dataset("T:\\stolt_all_aligned\\stolt-2385_0.jpg")
      #list_files(str(PATH / 'test/*.jpg'))
      #except tf.errors.InvalidArgumentError:
      #    test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))  
      #new_dataset = new_dataset.map(load_jpg)
      #new_dataset = new_dataset.batch(1)      
      
      """
      filenames = tf.constant(["T:\\stolt_all_aligned\\stolt-2385_0.jpg"])  
      #new_dataset = tf.data.Dataset(["T:\\stolt_all_aligned\\stolt-2385_0.jpg"])
      new_dataset = tf.data.Dataset(filenames,[test_input])
      #new_dataset = tf.data.Dataset.from_tensor_slices((filenames, load_jpg)
      new_dataset = new_dataset.map(load_jpg)
      
      #tf.data.Dataset(load_jpg("T:\\stolt_all_aligned\\stolt-2385_0.jpg"))
      test_input = next(iter(new_dataset.take(1)))
      """
      
      #tf.keras.utils.save_img("T:\\DFL-input_bababa.png", test_input)   #KERAS 2
      print(dir(tf.compat.v1.keras))
      print(dir(tf.compat.v1.keras.utils))
       
      # tf.compat.v1.keras.utils.save_img("T:\\DFL-input_bababa.png", test_input)  
      tf.keras.preprocessing.image.save_img("T:\\DFL-input_bababa.png", test_input)
      
     
      
      #test_input /= 255.0
      #plt.imshow(test_input/255.0)
      
      print(test_input)
      #prediction = generator(test_input, training=True)
      #prediction = generator(tf.expand_dims(test_input, axis=0), training=True)  
      prediction = generator(test_input[tf.newaxis, ...], training=True) #False) #True) #False)  
      #prediction = generator(test_input, training=True) #False) #True) #False)  
      #prediction = generator(test_input[:], training=True) #False)  
      print("PREDICTION:")
      print(prediction)
      ##### plt.imshow(prediction[0, ...]* 0.5 + 0.5)
      
      
      #tf.keras.utils.save_img("T:\\DFL-predicted_bababa.png", tf.squeeze(prediction)) #prediction)
      
      #tf.keras.preprocessing.image.save_img("T:\\DFL-input_bababa.png", test_input)  
      
      
      #gen_output = generator(inp[tf.newaxis, ...], training=False)
      
      # print(tf.expand_dims(test_input, axis=0))
      
      #tf.keras.utils.save_img("T:\\predicted_bababa.png", prediction)
      # tf.keras.utils.save_img("T:\\predicted_bababa.png",  tf.squeeze(prediction))
      # print(tf.squeeze(prediction))
      # plt.figure(figsize=(7, 5))  
      # plt.title("PREDICTED COLOR")
      # display_list = [test_input[0], test_input[0], prediction[0]]
      # title = ['Input Image', "Input", 'Predicted Image']  
      """
        for i in range(3):
          plt.subplot(1, 3, i+1)
          plt.title(title[i])
          # Getting the pixel values in the [0, 1] range to plot.
          if show:
            plt.imshow(display_list[i] * 0.5 + 0.5)    
            plt.axis('off')      
          #if name!=None: plt.savefig(name)
          plt.savefig("T:\gyzzzzzzzz.jpg")
          #input("Press [enter] to continue.")
        plt.pause(0.001)
        plt.show() #block = False) #TEsT
      """
      #print(tf.expand_dims(test_input, axis=0))
        
      #tf.keras.utils.save_img("T:\\predicted_bababa.png", prediction)
      #tf.keras.utils.save_img("T:\\predicted_bababa.png", tf.squeeze(prediction))
      #tf.keras.utils.save_img("T:\\KUR-DFL-predicted_bababa.png", tf.squeeze(prediction))
      tf.keras.preprocessing.image.save_img("T:\\KUR-DFL-predicted_bababa.png", tf.squeeze(prediction))

      k = input("Press EN ... ")
      
      #RESIZE
      
      height_new = 192
      width_new = 192
      return tf.squeeze(tf.image.resize(test_input, [height_new, width_new],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))

      
