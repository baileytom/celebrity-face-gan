from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import signal
import sys
import numpy as np
import os
import pathlib
import cv2

losses = []
accuracies = []
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
z_dim = 100
plt.switch_backend('agg')

# Take an image filename & return the normalized numpy array
def image_to_np(filename):
    image = cv2.imread(str(filename))
    image = cv2.resize(image, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    return image/127.5-1

# Definition of generator
def generator(z_dim):
    sx, sy = np.int(img_rows/4), np.int(img_cols/4)
    
    model = Sequential()
    model.add(Dense(256 * sx * sy, input_dim=z_dim))
    model.add(Reshape((sx, sy, 256)))
    model.add(Conv2DTranspose(
                128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(
                64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(
                3, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))
    z = Input(shape=(z_dim,))
    img = model(z)
    return Model(z, img)

# Definition of discriminator
def discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    img = Input(shape=img_shape)
    prediction = model(img)
    return Model(img, prediction)

# Create noisy labels
def noisy_labels(label, batch_size):
    mislabeled = batch_size // 10
    labels = []
    if label:
        labels = np.concatenate([
            np.random.normal(0.7, 1, batch_size-mislabeled),
            np.random.normal(0, 0.3, mislabeled)], axis=0)
    else:
        labels = np.concatenate([
            np.random.normal(0, 0.3, batch_size-mislabeled),
            np.random.normal(0.7, 1, mislabeled)], axis=0)
    return np.array(labels)

def train(X_train, epochs, batch_size, sample_interval):

    # Noisy labels effectively handicap the discriminator,
    # giving the generator a chance to learn.
    ones = noisy_labels(1, batch_size)
    zeros = noisy_labels(0, batch_size)

    for epoch in range(epochs):
        
        ind = np.random.randint(0, X_train.shape[0], batch_size)
        images = X_train[ind]

        # Generate images
        z = np.random.normal(0, 1, (batch_size, 100))
        images_gen = generator.predict(z)
        
        # Discriminator loss
        d_loss = discriminator.train_on_batch(images, ones)
        d_loss_gen = discriminator.train_on_batch(images_gen, zeros)
        d_loss = 0.5 * np.add(d_loss, d_loss_gen)

        # Generate images -- What if we used the same ones as before?
        z = np.random.normal(0, 1, (batch_size, 100))
        images_gen = generator.predict(z)

        # Generator loss
        g_loss = combined.train_on_batch(z, ones)

        print ('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        losses.append((d_loss[0], g_loss))
        accuracies.append(100*d_loss[1])
        
        if epoch % sample_interval == 0:
            sample_images(epoch)

def sample_images(epoch, image_grid_rows=4, image_grid_columns=4):
    plt.figure(figsize=(10,10))
    
    # Sample random noise
    z = np.random.normal(0, 1, 
              (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale images to 0-1
    gen_imgs = 0.5 * gen_imgs + 0.5
 
    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i+1)
        image = gen_imgs[i, :, :, :]

        try:
            image = np.reshape(image, [img_cols, img_rows, channels])
        except:
            image = np.reshape(image, [img_cols, img_rows])

        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()

    if not os.path.exists('./images/{}'.format(directory)):
        os.makedirs('./images/{}'.format(directory))
    filename = './images/{}/sample_{}.png'.format(directory, epoch)

    f = open('./images/{}/{}'.format(directory, word), 'w+')
    f.close()
    
    plt.savefig(filename)
    plt.close('all')

def process_source(root, directory):
    data_root = pathlib.Path('./{}'.format(root))
    image_root = data_root / 'train/{}/images'.format(directory)
    image_paths = list(image_root.glob('*.JPEG'))
    images = [image_to_np(image) for image in image_paths]
    return np.array(images)

def signal_handler(sig, frame):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    discriminator.save_weights('models/{}_d.h5'.format(directory))
    generator.save_weights('models/{}_g.h5'.format(directory))
    print("Saved weights.")
    sys.exit(0)
    
# Read the imagenet sources to easily select a set of images
def read_sources():
    # Get the list of words that correspond to each directory
    f = open('tiny-imagenet-200/words.txt', 'r')
    words = f.readlines()
    dir_map = {}
    for line in words:
        line = line.split("\t")
        directory = line[0]
        tags = line[1].strip().split(', ')
        dir_map[directory] = tags
    # Get the list of directories we have access to & reverse map every word
    word_map = {}
    data_root = pathlib.Path('./tiny-imagenet-200/train/')
    for child in data_root.iterdir():
        name = str(child.name)
        for word in dir_map[name]:
            word_map[word] = name
    return word_map

signal.signal(signal.SIGINT, signal_handler)

## Get the user's input

source_map = read_sources()
directory = None
word = None

print("\n\n\n\nThese are the tags you can target:\n\n")
print(list(source_map.keys()))

while(True):
    try:
        word = input("Pick one: ")
        directory = source_map[word]
        break
    except:
        print("That didn't work, try another one.")
data = process_source('tiny-imagenet-200', directory)

## Set up training

discriminator = discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(), metrics=['accuracy'])

try:
    discriminator.load_weights('models/{}_d.h5'.format(directory))
    print('Loaded discriminator weights.')
except:
    pass

generator = generator(z_dim)

try:
    generator.load_weights('models/{}_g.h5'.format(directory))
    print('Loaded generator weights.')
except:
    pass

z = Input(shape=(100,))
img = generator(z)

discriminator.trainable = False

prediction = discriminator(img)

combined = Model(z, prediction)
combined.compile(loss='binary_crossentropy', optimizer=Adam())

epochs = 1000000
batch_size = 32
sample_interval = 1000

## Train
train(data, epochs, batch_size, sample_interval)
