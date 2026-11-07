The UNet-from-scratch project is a clean and reproducible implementation of the U-Net architecture in deep neural networks. The U-Net architecture was introduced by Ronneberger and colleagues in 2015 to achieve high precision results in medical imaging.

This architecture is particularly effective in computer vision examples, especially in the field of medical image diagnostics and image translation. This project is built based on the core principles of the U-Net architecture and uses it for training a laboratory prototype.

The U-Net architecture consists of two main components. The first part of the network is composed of a series of convolutional layers and encoder layers. This part is crucial for extracting important information from the image. Different convolutional filters with different dimensions and accuracies are used in this part to ensure simultaneous information transfer to different levels of the image.

The second part of the network is the reconstruction part, which includes upsampling layers and deconvolution layers. In this part, the information obtained in the feature reduction process in the first part is returned and used to recover the necessary details of the image. Due to its similarity to the shape of the letter U, this architecture is known as U-Net.

The UNet-from-scratch project includes source code for building the U-Net network without using ready-made libraries. This project is a tutorial example that can be implemented for training the U-Net network in your own projects.
