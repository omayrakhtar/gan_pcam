"""Implements utility functions"""

from math import sqrt
from os.path import join
from typing import Dict, Optional

import attr
import numpy as np
from matplotlib import pyplot as plt

from data_wrangling.data_loader import DataLoader
from models.conditional_gan import GanModel


@attr.s(auto_attribs=True)
class TrainParam:
    """Represents the three components of a GAN model"""

    n_epochs: int
    batch_size: int
    latent_dim: int
    epoch_checkpoint: int
    n_summary_samples: int
    config: Dict[str, str]
    starting_epoch: Optional[int] = None

def trainer(gan_model: GanModel, data_loader: DataLoader, tarin_param: TrainParam):
    n_batches = int(data_loader.X.shape[0] / tarin_param.batch_size)
    dataset_indices = np.arange(data_loader.X.shape[0])
    epoch_dloss_r = []
    epoch_dloss_f = []
    epoch_gen_loss = []
    starting_epoch = 0 if tarin_param.starting_epoch is None else tarin_param.starting_epoch
    for epoch in range(starting_epoch, tarin_param.n_epochs):
        np.random.shuffle(dataset_indices)
        batch_dloss_r = []
        batch_dloss_f = []
        batch_gen_loss = []
        for batch in range(n_batches):
            batch_indices = get_batch_indices(
                dataset_indices=dataset_indices,
                batch=batch,
                batch_size=tarin_param.batch_size
            )
            X_real, labels_real, y_real = data_loader.generate_real_samples(indices=batch_indices, label_smoothing=True)
            dloss_r, _, _ = gan_model.discriminator.train_on_batch(X_real, [y_real, labels_real])
            
            X_fake, labels_fake, y_fake = data_loader.generate_fake_samples(
                generator=gan_model.generator,
                latent_dim=tarin_param.latent_dim,
                n_samples=tarin_param.batch_size
            )
            dloss_f, _, _ = gan_model.discriminator.train_on_batch(X_fake, [y_fake, labels_fake])
            
            z_input, z_labels = data_loader.generate_latent_points(
                latent_dim=tarin_param.latent_dim, n_samples=tarin_param.batch_size)
            y_gan = np.ones((tarin_param.batch_size, 1))
            gen_loss, _, _ = gan_model.gan.train_on_batch([z_input, z_labels], [y_gan, z_labels])
            
            print(f'>Batch:{batch+1} DR[{dloss_r:.3f}], DF[{dloss_f:.3f}] GL[{gen_loss:.3f}]\r', end="")
            batch_dloss_r.append(dloss_r)
            batch_dloss_f.append(dloss_f)
            batch_gen_loss.append(gen_loss)

        print(f'>Epoch:{epoch+1} DR[{np.mean(batch_dloss_r):.3f}], '
              f'DF[{np.mean(batch_dloss_f):.3f}] GL[{np.mean(batch_gen_loss):.3f}]')

        epoch_dloss_r.append(np.mean(batch_dloss_r))
        epoch_dloss_f.append(np.mean(batch_dloss_f))
        epoch_gen_loss.append(np.mean(batch_gen_loss))

        if (epoch+1) % tarin_param.epoch_checkpoint == 0:
            summarizer(
                epoch=epoch+1,
                gan_model=gan_model,
                data_loader=data_loader,
                tarin_param=tarin_param
            )
        
    return epoch_dloss_r, epoch_dloss_f, epoch_gen_loss

def get_batch_indices(dataset_indices: np.ndarray, batch: int, batch_size: int):
    start_index = batch * batch_size
    jump_index = batch * batch_size + batch_size 
    end_index = jump_index if jump_index < len(dataset_indices) else len(dataset_indices)
    return dataset_indices[start_index:end_index]

def summarizer(epoch: int, gan_model: GanModel, data_loader: DataLoader, tarin_param: TrainParam):
    X, _, _ = data_loader.generate_fake_samples(
        generator=gan_model.generator,
        latent_dim=tarin_param.latent_dim,
        n_samples=tarin_param.n_summary_samples
    )
    X = ((X * 127.5) + 127.5).astype('uint8')
    sample_sqrt = int(sqrt(tarin_param.n_summary_samples))
    plt.figure(figsize=(15, 15))
    for i in range(tarin_param.n_summary_samples):
        plt.subplot(sample_sqrt, sample_sqrt, 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, :])
    plt.savefig(join(tarin_param.config['output_path'], f'generated_plot_{epoch}.png'))
    plt.close()
    gan_model.generator.save(join(tarin_param.config['model_path'], f'g_model_{epoch}.h5'))
    gan_model.discriminator.save(join(tarin_param.config['model_path'], f'm_model_{epoch}.h5'))
