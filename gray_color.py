
#Importing packages
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import os
import time
import tensorflow as tf
from tensorflow import keras
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

from keras.layers import Conv2D, LeakyReLU, Input, Concatenate, Conv2DTranspose, MaxPooling2D, Flatten, Dense

import blip


# load the BLIP model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")





# defining model parameters
img_size = 128
dataset_split = 8000
image_dir = 'flickr'


# creating grayscale and color images
gs = []
rgb = []
for i, image_file in enumerate(os.listdir(image_dir)):
    if i >= dataset_split:
        break

    # loading color images
    rgb_image = cv2.imread(os.path.join(image_dir, image_file))
    rgb_image = cv2.resize(rgb_image, (img_size, img_size))
    
    # loading grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    #normalization
    gray_img_array = np.array(gray_image) / 255.0
    rgb_img_array = np.array(rgb_image) / 255.0

    # appending to dataset
    gs.append(gray_img_array)
    rgb.append(rgb_img_array)


# array conversion
gs = np.array(gs)
rgb = np.array(rgb)


# splitting train, test data
train_x, test_x, train_y, test_y = train_test_split(np.array(gs), np.array(rgb), test_size=0.2)








# defining generator
def generator_model():

    inputs = Input(shape=(img_size, img_size, 3))

    x1 = Conv2D(32, kernel_size=5, strides=1)(inputs)
    x1 = LeakyReLU()(x1)
    x1 = Conv2D(64, kernel_size=3, strides=1)(x1)
    x1 = LeakyReLU()(x1)
    x1 = Conv2D(64, kernel_size=3, strides=1)(x1)
    x1 = LeakyReLU()(x1)

    x2 = Conv2D(64, kernel_size=5, strides=1)(x1)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(128, kernel_size=3, strides=1)(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(128, kernel_size=3, strides=1)(x2)
    x2 = LeakyReLU()(x2)

    x3 = Conv2D(128, kernel_size=5, strides=1)(x2)
    x3 = LeakyReLU()(x3)
    x3 = Conv2D(256, kernel_size=3, strides=1)(x3)
    x3 = LeakyReLU()(x3)
    x3 = Conv2D(256, kernel_size=3, strides=1)(x3)
    x3 = LeakyReLU()(x3)

    x4 = Conv2D(256, kernel_size=5, strides=1)(x3)
    x4 = LeakyReLU()(x4)
    x4 = Conv2D(512, kernel_size=3, strides=1)(x4)
    x4 = LeakyReLU()(x4)
    x4 = Conv2D(512, kernel_size=3, strides=1)(x4)
    x4 = LeakyReLU()(x4)

    x5 = Conv2D(512, kernel_size=3, strides=1, activation='tanh', padding='same')(x3)

    x5 = Concatenate()([ x5, x4 ])
    x6 = Conv2DTranspose(512, kernel_size=3, strides=1, activation='relu')(x5)
    x6 = Conv2DTranspose(512, kernel_size=3, strides=1, activation='relu')(x6)
    x6 = Conv2DTranspose(256, kernel_size=5, strides=1, activation='relu')(x6)

    x7 = Concatenate()([ x7, x3 ])
    x8 = Conv2DTranspose(256, kernel_size=3, strides=1, activation='relu')(x7)
    x8 = Conv2DTranspose(256, kernel_size=3, strides=1, activation='relu')(x8)
    x8 = Conv2DTranspose(128, kernel_size=5, strides=1, activation='relu')(x8)

    x9 = Concatenate()([ x9, x2 ])
    x10 = Conv2DTranspose(128, kernel_size=3, strides=1, activation='relu')(x9)
    x10 = Conv2DTranspose(128, kernel_size=3, strides=1, activation='relu')(x10)
    x10 = Conv2DTranspose(64, kernel_size=5, strides=1, activation='relu')(x10)

    x11 = Concatenate()([ x11, x1 ])
    x12 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu')(x11)
    x12 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu')(x12)
    x12 = Conv2DTranspose(3, kernel_size=5, strides=1, activation='relu')(x12)

    model = tf.keras.models.Model(inputs, x10)

    #returning model
    return model







# discriminiator model build
def discriminator_model():
    model = tf.keras.models.Sequential()
    
    model.add(Conv2D(64, kernel_size=7, strides=1, activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(Conv2D(64, kernel_size=7, strides=1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=5, strides=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=5, strides=1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(512, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv2D(512, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # returning model
    return model









# defining optimizer
generator_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)

# loading discriminator
discriminator = discriminator_model()

# compiling disc
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])


# loading generator
generator = generator_model()

# compiling gen
generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer, metrics=['accuracy'])


# printing gen model
generator.summary()

# printing disc model
discriminator.summary()


# params for model
batch_size = 10
total_batches = len(train_x) // batch_size
epochs = 10

# running train model
for epoch in range(epochs):
    # batch size
    for i in range(0, len(train_x), batch_size):
        # train disc
        idx = np.random.randint(0, len(train_y), batch_size)
        real_imgs = train_y[idx]
        labels = np.ones((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_imgs, labels)

        # train gen
        fake_imgs = generator.predict(train_x[i:i+batch_size])
        labels = np.zeros((batch_size, 1))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        g_loss = generator.train_on_batch(train_x[i:i+batch_size], real_imgs)

    # print train loss
    print(f"{epoch} [Disc loss: {d_loss[0]}, Accuracy: {d_loss[1] * 100:.2f}%] [Gen loss: {g_loss}]")



# predicting images
y = generator.predict(test_x)

yy = generator(test_x)


# printing 100 images
for i in range(0, 100):
    
    plt.figure(figsize=(10,10))
    

    gs_image = plt.subplot(3,3,1)
    gs_image.set_title('Grayscale Input', fontsize=13)
    plt.imshow(test_x[i], cmap='gray')

    ou_image = plt.subplot(3,3,2)
    ou_image.set_title('Model Output', fontsize=13)
    plt.imshow(y[i])

    re_image = plt.subplot(3,3,3)
    re_image.set_title('Ground Truth', fontsize=13)
    plt.imshow(test_y[i], cmap="viridis")


    # generating captions
    inputs = blip_processor(test_y[i], return_tensors="pt")
    caption = blip_processor.decode(blip_model.generate(**inputs)[0], skip_special_tokens=True)

    # adding captions as title
    plt.suptitle(caption, fontsize=16)


    # saving figure
    plt.savefig('{}.png'.format(i+1))

    # displaying image
    plt.show()



