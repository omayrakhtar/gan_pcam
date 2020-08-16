"""Implements conditional GAN with auxiliary classifier"""

from typing import Tuple, Optional

import attr
from keras import layers
from keras.initializers import RandomNormal
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Dense, Dropout, Embedding, Flatten,
                          Input, LeakyReLU, Reshape)
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot
from numpy import expand_dims, ones, zeros
from numpy.random import randint, randn


class CGanGenerator:
    """Conditional GAN generator with class label as embedding layer"""
    
    def __init__(
            self,
            latent_dimension: int = 100,
            n_classes: int = 2,
            filter_list: Optional[int] = None,
            output_channels: int = 3
        ) -> None:
        self._latent_dimension = latent_dimension
        self._n_classes = n_classes
        self._weight_initializer = RandomNormal(stddev=0.02)
        self._filter_list = [512, 512, 512] if filter_list is None else filter_list
        self._output_channels = 3
    
    def __call__(self):
        init = RandomNormal(stddev=0.02)
        in_label = Input(shape=(1,))
        # embedding for categorical input
        label_out = Embedding(self._n_classes, 50)(in_label)
        label_out = Dense(6 * 6, kernel_initializer=init)(label_out)
        label_out = Reshape((6, 6, 1))(label_out)
        in_lat = Input(shape=(self._latent_dimension,))
        gen = Dense(1024 * 6 * 6, kernel_initializer=init)(in_lat)
        gen = Activation('relu')(gen)
        gen = Reshape((6, 6, 1024))(gen)
        # merge image gen and label
        merge = Concatenate()([gen, label_out])
        # upsample
        self._filter_list.append(self._output_channels)
        for i, filters in enumerate(self._filter_list):
            conv_out = self._get_conv_transpose_block(
                conv_in=merge if i == 0 else conv_out,
                filters=filters,
                activation='tanh' if i == 3 else 'relu',
                batch_normalization=False if i == 3 else True
            )
        model = Model([in_lat, in_label], conv_out)
        return model

    def _get_conv_transpose_block(
            self,
            conv_in: layers,
            filters: int,
            activation: str,
            batch_normalization: bool
        ) -> layers:
        conv_out = Conv2DTranspose(
            filters=filters,
            kernel_size=(5,5),
            strides=(2,2),
            padding='same',
            kernel_initializer=self._weight_initializer
        )(conv_in)
        conv_out = BatchNormalization()(conv_out) if batch_normalization else conv_out
        conv_out = Activation(activation)(conv_out)
        return conv_out


class CGanDiscriminator:
    """Conditional GAN discriminator with auxiliary classifier"""
    
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (96,96,3),
            n_classes: int = 2,
            filter_list: Optional[int] = None
        ) -> None:
        self._input_shape = input_shape
        self._n_classes = n_classes
        self._weight_initializer = RandomNormal(stddev=0.02)
        self._filter_list = [64, 128, 256, 512, 512] if filter_list is None else filter_list
    
    def __call__(self) -> Model:
        in_image = Input(shape=self._input_shape)
        for i, filters in enumerate(self._filter_list):
            conv_out = self._get_conv_block(
                conv_in = in_image if i == 0 else conv_out,
                strides=(2,2),
                filters=filters,
                batch_normalization=False if i == 0 else True
            )
            conv_out = self._get_conv_block(
                conv_in = conv_out,
                strides=(1, 1),
                filters=filters,
                batch_normalization=True
            )
        flattened_out = Flatten()(conv_out)
        # real/fake output
        auth_label = Dense(1, activation='sigmoid')(flattened_out)
        # class label output (sigmoid can also be used since there are only two classes)
        class_label = Dense(self._n_classes, activation='softmax')(flattened_out)
        # define model
        model = Model(in_image, [auth_label, class_label])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

    def _get_conv_block(
            self,
            conv_in: layers,
            strides: Tuple[int, int],
            filters: int,
            batch_normalization: bool
        ) -> layers:
        conv_out = Conv2D(
            filters=filters,
            kernel_size=(3,3),
            strides=strides,
            padding='same',
            kernel_initializer=self._weight_initializer
        )(conv_in)
        conv_out = BatchNormalization()(conv_out) if batch_normalization else conv_out
        conv_out = LeakyReLU(alpha=0.2)(conv_out)
        conv_out = Dropout(0.5)(conv_out)
        return conv_out


class CGan:
    """Conditional GAN constructed by combining both generator and discriminator"""

    def __init__(self, generator: Model, discriminator: Model) -> None:
        self._d_model = discriminator
        self._g_model = generator

    def __call__(self) -> Model:
        self._d_model.trainable = False
        gan_output = self._d_model(self._g_model.output)
        gan_model = Model(self._g_model.input, gan_output)
        opt = Adam(lr=0.0002, beta_1=0.5)
        gan_model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return gan_model

@attr.s(auto_attribs=True)
class GanModel:
    """Represents the three components of a GAN model"""
    
    generator: Model
    discriminator: Model
    gan: Model
