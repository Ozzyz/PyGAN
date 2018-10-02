from keras.datasets import mnist
from keras.layers import Input, Dense, BatchNormalization, Reshape, Flatten, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

def build_generator(input_shape, img_shape):
    # Input for generator is noise vector
    inputs = Input(shape=input_shape)

    x = Dense(units=256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(units=512, activation='relu')(x)
    x = BatchNormalization()(x)

    img_vector_length = reduce(lambda x, y: x*y, img_shape)
    x = Dense(units=img_vector_length, activation='relu')(x)
    # Reshape back into original image shape
    outputs = Reshape(img_shape)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model


def build_discriminator(img_shape):
    inputs = Input(shape=img_shape)

    x = Flatten()(inputs)
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=512, activation='relu')(x)
    prediction = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def stack_gan(input_shape, generator, discriminator):
    gan_input = Input(shape=input_shape)
    gen_out = generator(gan_input)
    disc_out = discriminator(gen_out)
    
    net = Model(gan_input, disc_out)
    net.compile(loss='binary_crossentropy', optimizer=Adam())
    return net

def train_gan(gan, x_train, generator, discriminator, num_epochs=2000, batch_size=64):
    for epoch in range(num_epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        batch = x_train[idx]
        discriminator_loss = train_discriminator(batch, generator, discriminator)

        noise = np.random.normal(0, 1, (batch_size, 100))
        generator_loss = gan.train_on_batch(noise, np.ones(batch_size))

        print(f"Gen loss: {generator_loss}, Disc loss: {discriminator_loss}")


def train_discriminator(real_images, generator, discriminator):

    noise = np.random.normal(0, 1, (len(real_images), 100))
    generator_images = generator.predict(noise)

    # Train the discriminator on real and fake images

    batch_size = real_images.shape[0]
    train_x = np.concatenate((generator_images, real_images), axis=0)
    train_y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)), axis=0)
    
    discriminator_loss = discriminator.train_on_batch(train_x, train_y)
    return discriminator_loss



if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_train /= 255

    img_shape = (28, 28)
    input_shape = (100,)
    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    generator = build_generator(input_shape=input_shape, img_shape=img_shape)
    discriminator = build_discriminator(img_shape=img_shape) 
    
    gan = stack_gan(input_shape, generator, discriminator)

    train_gan(gan, x_train, generator, discriminator)
