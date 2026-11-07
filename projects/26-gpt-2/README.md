<div align="center">
    <img src="https://github.com/user-attachments/assets/c9771160-380c-441f-baf2-cb8845eb6072" alt="Logo" width="" height="200">
  </a>

<h1 align="center"></h1>
</div>

# GPT-2: Overview
GPT-2 (Generative Pre-trained Transformer 2) is a large language model developed by OpenAI in 2019. It is the successor to GPT-1 and is based on the Transformer architecture (specifically the decoder-only variant). GPT-2 was notable for its ability to generate coherent and contextually relevant text, making it a breakthrough in natural language processing (NLP).

**Key Features of GPT-2:**
   * **Model Size Variants: GPT-2 was released in multiple sizes:**

        * **Small** (117M parameters)

        * **Medium** (345M parameters)

        * **Large** (774M parameters)

        * **Extra Large (XL)** (1.5B parameters)

  * **Unsupervised Learning**: Pre-trained on a massive corpus of internet text (WebText) in a self-supervised manner (predicting the next word in a sequence).

  * **Fine-Tuning Capability**: Could be fine-tuned for specific tasks (e.g., translation, summarization).

  * **Controversy**: Initially, OpenAI withheld the full 1.5B model due to concerns about misuse (e.g., fake news generation), but later released it.

**Architecture**:
  * Based on **Transformer decoder blocks** (no encoder).

  * Uses **masked self-attention** (causal attention) to prevent looking ahead in the sequence.

  * Trained using **next-word prediction** (autoregressive modeling).

# Dataset
## [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
The TinyStories dataset is a synthetic, simplified text corpus designed for training and evaluating small language models. It was introduced to help researchers experiment with lightweight models that can still generate meaningful text.

**Key Features of TinyStories**:
 * **Simple Language**: Contains short stories with basic vocabulary and grammar, making it easier for small models to learn.

 * **Focused on Coherence**: Designed to test whether small models can maintain narrative consistency (unlike random text generation).

 * **Use Case**: Helps in studying how small transformers (1M–100M parameters) perform compared to large models like GPT-2.

**Why TinyStories?**
 * Large models like GPT-2 require massive computational resources.

 * TinyStories allows testing **fundamental language understanding** without needing huge models.

 * Useful for **educational and research purposes** in low-resource settings.

**Example TinyStories Text**:

<details>
<summary>📖 The Story of Max the Clever Dog </summary>
    
>Once upon a time, there was a clever little dog named Max. Max loved to run and play with his friends in the park. One day, Max was running very fast when he fell and hurt his knee.
Max went to his friend, the wise old owl, and said, "Owl, my knee hurts. What can I do?" The owl thought for a moment and said, "Max, you should test your knee. Try to walk slowly and see if it still hurts."
So Max tested his knee by walking slowly. At first, it hurt a little, but soon Max felt better. He said, "Thank you, Owl, for your help. Now I can play with my friends again."
Max was so happy that he could play with his friends without pain. He learned that sometimes, it was good to slow down and listen to his body. And Max and his friends played happily in the park ever after.

</details>

**the simple structure, making it ideal for small models.**

Here is [Dataset Analysis](https://github.com/ahmadrezabaqerzade/GPT2/blob/main/Analyze/EDA.pdf)
