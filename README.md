# Synthetic histopathology data generation using Conditional Generative Adversarial Network (cGAN)
This project uses conditional GAN with auxiliary classifier[1] to generate synthetic pathology data.
The dataset used in this project is called PatchCamelyon or PCAM[2]. PCam dataset, derived from
CAMELYON16[3], contains over two hundred thousand 96x96 breast histopathology slide image patches.
The patches contain both clinically significant metastatic cancer and healthy cells.
 
[1] Odena, A., Olah, C., & Shlens, J. (2017, July). Conditional image synthesis with auxiliary
classifier gans. In International conference on machine learning (pp. 2642-2651).

[2] Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M. (2018, September). Rotation
equivariant CNNs for digital pathology. In International Conference on Medical image computing and
computer-assisted intervention (pp. 210-218). Springer, Cham.

[3] Bejnordi, B. E., Veta, M., Van Diest, P. J., Van Ginneken, B., Karssemeijer, N., Litjens, G.,
... & Geessink, O. (2017). Diagnostic assessment of deep learning algorithms for detection of lymph
node metastases in women with breast cancer. Jama, 318(22), 2199-2210.

# Patch Samples
![alt text](https://github.com/omayrkhan/gan_pcam/blob/master/images/real-synthetic.png)


# Discriminator Architecture
![alt text](https://github.com/omayrkhan/gan_pcam/blob/master/images/discriminator.png)


# Generator Architecture
![alt text](https://github.com/omayrkhan/gan_pcam/blob/master/images/generator.png)


# GAN Architecture
![alt text](https://github.com/omayrkhan/gan_pcam/blob/master/images/gan.png)
