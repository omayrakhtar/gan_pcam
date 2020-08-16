"""Data loading functions"""

from os import listdir
from os.path import isfile, join
from typing import List, Sequence, Tuple, Union

import numpy as np
from keras.models import Model
from PIL import Image


class DataLoader:
    """Implements data loading and manipulation functions"""
    
    def __init__(
            self,
            data_path: str,
            sample_types: Sequence = ['positive', 'negative']
        ) -> None:
        self._data_path = data_path
        self.X, self.labels = self._load_sample_data(sample_types=sample_types)
    
    def _shuffle_samples(self):
        perm = np.random.permutation(len(self.X))
        self.X = self.X[perm]
        self.labels = self.labels[perm]

    def _load_sample_data(self, sample_types: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        images: List = []
        labels: List = []
        sample_paths = [join(self._data_path, sample_type) for sample_type in sample_types]
        for sample_path in sample_paths:
            image_paths = [f for f in listdir(sample_path) if isfile(join(sample_path, f))]
            for img_path in image_paths:
                img = Image.open(join(sample_path, img_path))
                images.append(np.array(img))
                labels.append(1 if 'positive' in sample_path else 0)
        return np.array(images), np.array(labels)

    def _get_normalized_samples(self):
        X = self.X.astype('float32')
        X = (X - 127.5) / 127.5
        return X

    def generate_real_samples(
            self,
            indices: np.ndarray,
            normalize: bool = True,
            label_smoothing: bool = False,
            class_label: bool = True,
        ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.array, np.array]]:
        """Generates a batch of real samples from the dataset"""
        X_real = self._get_normalized_samples()[indices] if normalize else self.X[indices]
        labels = self.labels[indices]
        n_samples = len(indices)
        y = np.ones((n_samples, 1))
        if label_smoothing:
            y -= np.random.uniform(0, 0.1, n_samples).reshape((n_samples, 1))
        if not class_label: 
            return X_real, y
        return X_real, labels, y 

    def generate_latent_points(
            self,
            latent_dim: int,
            n_samples: int,
            n_classes: int = 2,
            class_label: bool = True
        ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Generates a batch of latent vectors of random points"""
        x_input = np.random.randn(latent_dim * n_samples)
        z_input = x_input.reshape(n_samples, latent_dim)
        labels = np.random.randint(0, n_classes, n_samples)
        if not class_label:
            return z_input
        return z_input, labels

    def generate_fake_samples(
            self,
            generator: Model,
            latent_dim: int,
            n_samples: int,
            class_label: bool = True
        ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.array, np.array]]:
        """Generates a batch of fake samples from latent vectors using generator model"""
        z_input, labels = self.generate_latent_points(latent_dim, n_samples)
        images = generator.predict([z_input, labels] if class_label else z_input)
        y = np.zeros((n_samples, 1))
        if not class_label: 
            images, y
        return images, labels, y 
