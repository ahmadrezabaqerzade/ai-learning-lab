# Language Modeling

In this project, I initially attempted to reduce perplexity on the **WikiText2** dataset as much as possible using LSTM and simple regularizations, including embedding dropout, weight decay and dropout incorporated into the LSTM.

The hyperparameters were determined empirically through multiple training iterations and observing the obtained results, aiming to find their optimal values.

In the second step, I tried to significantly reduce perplexity on the WikiText2 dataset by employing the techniques described in the research paper **"Regularizing and Optimizing LSTM Language Models"**, known as **AWD-LSTM (ASGD Weight-Dropped LSTM)**. Each regularization technique was implemented separately and then combined with the previous technique. The following list presents the implemented techniques:

**1. Weight-Dropped LSTM**
**2. NTASGD optimizer**
**3. Combination of Weight Drop and NTASGD**
**4. Variational Dropout**
**5. Combination of Weight Drop, NTASGD, and Variational Dropout**
**6. Embedding Dropout**
**7. Combination of WD, NTASGD, VD, and ED**
**8. Weight Tying**
**9. Combination of WD, NTASGD, VD, ED, and WT**

After going through all these steps, I attempted to select the best method and apply the best combination of techniques. The combination of the best techniques was determined based on experiments and multiple training iterations. Although not all techniques, including sentence length variation, were implemented in this project, I tried to incorporate most of the mentioned techniques from the research paper to improve the results.

# Dependencies:
* Python: 3.10.12
* torch: 2.0.1+cu118
* torchtext: 0.15.2+cpu
* numpy: 1.23.5
* pandas: 1.5.3
* tqdm: 4.66.1
# Using LSTM:
In this section, I attempted to reduce the perplexity as much as possible on the dataset using LSTM, which includes three types of regularization. Below, you can see the adjusted hyperparameters:

* Learning rate: 0.5
* Weight decay: 1e-6
* Batch size: 20
* Number of epochs: 40
* Hidden size: 400
* Embedding dimension: 380
* Dropout rate (in LSTM layers): 0.2

Below, you can see the final result:

|parameters | perplexity(on validation)| perplexity(on test)|
| :--: | :--: | :--: |
|24.8M      | 128.3                    | 121.7              |

## Generate:
In this year **_, as he was in the first game of the season , and he was named the team ’ s captain ._**

This recommended approach helps prevent potential issues related to computational graph tracking  **_, as he was in the first game of the season , and he was named the team ’ s captain ._**   

I think the movie **_was good._**

The text generation results indicate that the model understands some of the initial relationships between words and their meanings. However, it doesn't adhere well to grammatical rules, and the model still has weaknesses in generating high-quality text.

# Using AWD-LSTM:

In this section, after multiple experiments with various techniques, a combination of techniques was used, which resulted in the best performance on the dataset. Due to the application of numerous regularizations, the training process improved, and the learning rate increased significantly as a result of using these techniques and gradient clipping. You can see the final result below:

|parameters | perplexity(on validation)| perplexity(on test)|
| :--: | :--: | :--: |
|31.75M      | 80.27                    | 77.11              |

## Generate:

I think this movie **_is the most beautiful thing i’ve ever seen ._**


The text generation results have improved compared to the previous model, and the model has also learned the rules better. However, it can be said that the model still cannot generate text at an excellent level. Text generation models using LSTM have less learning capability and capacity compared to other modern models such as transformers. Generally, these networks are not suitable for tasks with larger dimensions, and other approaches based on different networks should be used instead.

