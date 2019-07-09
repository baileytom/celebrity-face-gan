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

# Disables warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.switch_backend('agg')

# Set these based on your image size. Rows & cols should be a multiple of 4
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
z_dim = 100

# Take an image filename & return the normalized numpy array
def image_to_np(filename):
    image = cv2.imread(str(filename))
    image = cv2.resize(image, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
def noisy_labels(label, batch_size, ratio=0.4):
    mislabeled = batch_size // 10
    labels = []
    if label:
        labels = np.concatenate([
            np.random.normal(1-ratio, 1, batch_size-mislabeled),
            np.random.normal(0, ratio, mislabeled)], axis=0)
    else:
        labels = np.concatenate([
            np.random.normal(0, ratio, batch_size-mislabeled),
            np.random.normal(1-ratio, 1, mislabeled)], axis=0)
    return np.array(labels)

# Train the model
def train(X_train, epochs, batch_size, sample_interval, save_interval):
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

        print ('%s %d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (label, epoch, d_loss[0], 100*d_loss[1], g_loss))

        if epoch % save_interval == 0:
            save_weights()
        
        if epoch % sample_interval == 0:
            sample_images(epoch)

# Save a 4x4 grid of images
def save_images(images, path, filename):
    plt.figure(figsize=(10,10))
    for i in range(16):
        image = images[i]
        plt.subplot(4, 4, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + '/' + filename
    plt.savefig(filename)
    plt.close('all')

# Generate & save 16 images
def sample_images(epoch):
    # Sample random noise
    z = np.random.normal(0, 1, 
              (16, z_dim))
    # Generate images from random noise
    gen_imgs = generator.predict(z)
    # Rescale images to 0-1
    gen_imgs = 0.5 * gen_imgs + 0.5
    # Save
    save_images(list(gen_imgs),
                   './images/{}'.format(directory),
                   '{}_{}'.format(label.replace(" ", ""), epoch))
    
# Load training data, and save a sample of it to the directory
def process_source(root, directory):
    data_root = pathlib.Path('./{}'.format(root))
    image_root = data_root / 'train/{}/images'.format(directory)
    image_paths = list(image_root.glob('*.JPEG'))
    images = [image_to_np(image) for image in image_paths]
    save_images(images[:16],
                   './images/{}'.format(directory),
                   'example')
    return np.array(images)

# Catches ctrl-c to gracefully save weights before exit
def signal_handler(sig, frame):
    save_weights()
    sys.exit(0)

def save_weights():
    if not os.path.exists('./models'):
        os.makedirs('./models')
    discriminator.save_weights('models/{}_d.h5'.format(directory))
    generator.save_weights('models/{}_g.h5'.format(directory))
    print("\nSaved weights.")
    
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

def select_data():
    source_map = read_sources()
    os.system('clear')
    print("\nThese are the tags you can target:\n")
    print(list(source_map.keys()))
    while(True):
        try:
            label = input("\nPick one: ")
            directory = source_map[label]
            break
        except:
            print("That didn't work, try another one.")
    return directory, label

#========================================================================================

signal.signal(signal.SIGINT, signal_handler)

directory, label = select_data()
data = process_source('tiny-imagenet-200', directory)


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

z = Input(shape=(z_dim,))
image = generator(z)
discriminator.trainable = False
prediction = discriminator(image)

combined = Model(z, prediction)
combined.compile(loss='binary_crossentropy', optimizer=Adam())

epochs = 1000000
batch_size = 256
sample_interval = 100
save_interval = 50000

## Train
train(data, epochs, batch_size, sample_interval, save_interval)
