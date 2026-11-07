# Image Captioning

In this project, I aim to perform image captioning using convolutional and recurrent neural networks and obtain a textual description of an image from the model. Initially, I will attempt a simple approach by using **ResNet50** to convert the image into a rich feature vector and an **LSTM** network to generate text. In the first section, the ResNet50 network is used, and it **is frozen (except for the last layer, which is a linear layer that needs to be trained)**. In the second section of the project, I will add attention to this network and make some modifications to the network and also i use **Resnet101** as convlutional part. The Flickr8k dataset is used for training the models, which consists of **8,000** images, each with **5 captions**. In the first section, only the tokens from the training data are used to create the vocabulary. However, in the second section, where attention is used, the tokens include all the tokens from the dataset, which are used in creating the vocabulary.

# Dependencies:
**Version of libraries:**
* **numpy : 1.23.5**
* **torch : 2.1.0+cu118**
* **torchtext : 0.16.0+cpu**
* **torchvision : 0.16.0+cu118**
* **tqdm : 4.66.1**

# Using CNN+LSTM

* **embedding-dim: 256**
* **hidden-dim: 256**
* **num-layers: 2**
* **embedding-dropout: 0.3**
* **lstm-dropout: 0.3**
* **batch-size: 128**
* **learning rate: 0.9**
* **weight decay: 1e-6**
* **optimizer : SGD**


**Note that the training loop is configured in such a way that if the validation**
**loss does not improve in each epoch, the learning rate is reduced to 0.2, and**
**the optimizer continues its work with the new learning rate.**

<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2003-48-12.png"  title="result">


# Generate Result
<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2003-52-40.png"  title="result">

<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2003-52-49.png"  title="result">

# Using CNN+LSTM+Attention
* **attention-dim : 256**
* **embed-dim : 512**
* **decoder-dim : 512**
* **dropout : 0.5**
* **batch-size: 32**
* **learning rate for decoder: 4e-4**
* **weight decay for decoder: 1e-5**
* **optimizer : Adam**

**Note that the training loop is configured in such a way that if the validation**
**loss does not improve in each epoch(if epoch number greater than 10), the**
**learning rate is reduced to 0.8, and the optimizer continues its work with the**
**new learning rate.**

<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2004-02-23.png"  title="result">

# Generate Result
<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2004-06-51.png"  title="result">

<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2004-07-01.png"  title="result">

<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2004-07-09.png"  title="result">

<img src="https://github.com/ahmadrezabaqerzade/ImageCaptioning/blob/main/images/Screenshot%20from%202023-11-01%2004-07-19.png"  title="result">




















