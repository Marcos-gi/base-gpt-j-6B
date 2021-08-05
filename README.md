---
language:
- en
tags:
- pytorch
- causal-lm
license: apache-2.0
datasets:
- the Pile
---

# GPT-J 6B

## Model Description

GPT-J 6B is a transformer model designed using Ben Wang's [Jax implementation of the GPT-3 architecture](https://github.com/kingoflolz/mesh-transformer-jax/). GPT-J refers to the class of models, while 6B represents the number of parameters of this particular pre-trained model.

| Hyperparameter    | Value  | 
|-------------------|--------|
| n_parameters      | 6,053,381,344 |
| n_layers          | 28*    |
| d_model           | 4,096  |
| d_ff              | 16,384 |
| n_heads           | 16     |
| d_head            | 256    |
| n_ctx             | 2,048  |
| n_vocab           | 50,257 (same tokenizer as GPT-2/3)  |
| position encoding | [Rotary position encodings (RoPE)](https://arxiv.org/abs/2104.09864) |
| RoPE dimensions   | [64](https://github.com/kingoflolz/mesh-transformer-jax/blob/f2aa66e0925de6593dcbb70e72399b97b4130482/mesh_transformer/layers.py#L223) |

`*` each layer consists of one feedforward block and one self attention block

The model consists of 28 layers with a model dimension of 4096, and a feedforward dimension of 16384. The model
dimension is split into 16 heads, each with a dimension of 256. Rotary position encodings (RoPE) was applied to 64
dimensions of each head. The model is trained with a tokenization vocabulary of 50257, using the same set of BPEs as
GPT-2/GPT-3.

## Training data

GPT-J 6B was trained on the [Pile](pile.eleuther.ai), a large scale curated dataset created by EleutherAI for the purpose of training this model.

## Training procedure

This model was trained for [number] billion tokens over [number] steps. It was trained as a masked autoregressive language model, using cross-entropy loss.

## Intended Use and Limitations

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a prompt.

### How to use

You can use this model directly with a pipeline for text generation. This example generates a different sequence each time it's run:

```py
>>> from transformers import pipeline
>>> generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')
>>> generator("EleutherAI has", do_sample=True, min_length=50)

[{'generated_text': 'EleutherAI has made a commitment to create new software packages for each of its major clients and has'}]
```

### Limitations and Biases

GPT-J was trained as an autoregressive language model. This means that its core functionality is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. When prompting GPT-J with er that the statistically most likely next token is often not the same thing as the most accurate, 

GPT-J was trained on the Pile, a dataset known to contain profanity, lewd, and otherwise abrasive language. Depending on your usecase GPT-Neo may produce socially unacceptable text. See Sections 5 and 6 of the Pile paper for a more detailed analysis of the biases in the Pile.

As with all language models, it is hard to predict in advance how GPT-Neo will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results. 

## Eval results

Models roughly sorted by performance, or by FLOPs if not available.

|  Model          | Public  | Training FLOPs | LAMBADA PPL ↓ | LAMBADA Acc ↑ | Winogrande ↑ | Hellaswag ↑ | PIQA ↑    | Dataset Size (GB) |
|-----------------|---------|----------------|---            |---            |---           |---          |---        |-------------------|
| Chance          | ✔       | 0              | ~a lot        | ~0%           | 50%          | 25%         | 25%       | 0                 |
| GPT-3-Ada‡      | ✘       | -----          | 9.95          | 51.6%         | 52.9%        | 43.4%       | 70.5%     | -----             |
| GPT-2-1.5B      | ✔       | -----          | 10.63         | 51.21%        | 59.4%        | 50.9%       | 70.8%     | 40                |
| GPTNeo-1.3B‡    | ✔       | 3.0e21         | 7.50          | 57.2%         | 55.0%        | 48.9%       | 71.1%     | 825               |
| Megatron-2.5B*  | ✘       | 2.4e21         | -----         | 61.7%         | -----        | -----       | -----     | 174               |
| GPTNeo-2.7B‡    | ✔       | 6.8e21         | 5.63          | 62.2%         | 56.5%        | 55.8%       | 73.0%     | 825               |
| GPT-3-1.3B*‡    | ✘       | 2.4e21         | 5.44          | 63.6%         | 58.7%        | 54.7%       | 75.1%     | ~800              |
| GPT-3-Babbage‡  | ✘       | -----          | 5.58          | 62.4%         | 59.0%        | 54.5%       | 75.5%     | -----             |
| Megatron-8.3B*  | ✘       | 7.8e21         | -----         | 66.5%         | -----        | -----       | -----     | 174               |
| GPT-3-2.7B*‡    | ✘       | 4.8e21         | 4.60          | 67.1%         | 62.3%        | 62.8%       | 75.6%     | ~800              |
| Megatron-11B†   | ✔       | 1.0e22         | -----         | -----         | -----        | -----       | -----     | 161               |
| **GPT-J-6B**‡   | ✔       | **1.5e22**     | **3.99**      | **69.7%**     | **65.3%**    | **66.1%**   | **76.5%** | **825**           |
| GPT-3-6.7B*‡    | ✘       | 1.2e22         | 4.00          | 70.3%         | 64.5%        | 67.4%       | 78.0%     | ~800              |
| GPT-3-Curie‡    | ✘       | -----          | 4.00          | 69.3%         | 65.6%        | 68.5%       | 77.9%     | -----             |
| GPT-3-13B*‡     | ✘       | 2.3e22         | 3.56          | 72.5%         | 67.9%        | 70.9%       | 78.5%     | ~800              |
| GPT-3-175B*‡    | ✘       | 3.1e23         | 3.00          | 76.2%         | 70.2%        | 78.9%       | 81.0%     | ~800              |
| GPT-3-Davinci‡  | ✘       | -----          | 3.0           | 75%           | 72%          | 78%         | 80%       | -----             |

`*` represents evaluation numbers reported by their respective authors, all other numbers are provided by
running the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/) either with the released
weights or with API access. Due to subtle implementation differences as well as different zero shot task framing, these
might not be directly comparable. See [this blog post](https://www.eleuther.ai/research-log/gpt3-model-sizes/) for more
details.

`†` The Megatron-11B model provides no comparable metrics, and several implementations using the released weights do not
reproduce the generation quality and evaluations. (see [1](https://github.com/huggingface/transformers/pull/10301)
[2](https://github.com/pytorch/fairseq/issues/2358) [3](https://github.com/pytorch/fairseq/issues/2719))
Thus, evaluation was not attempted.

`‡` These models have been trained with data which contains possible test set contamination. The OpenAI GPT-3 models
failed to deduplicate training data for certain test sets, while the GPT-Neo models as well as this one is
trained on The Pile, which has not been deduplicated against any test sets.

### Down-Stream Applications

TBD

## Citation and Related Info

### BibTeX entry

To cite this model:
```
@misc{gpt-j,
  author = {Wang, Ben and Komatsuzaki, Aran},
  title = {{GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model}},
  howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
  year = 2021,
  month = May
}
```


To cite the codebase that trained this model:
```
@misc{mesh-transformer-jax,
  author = {Wang, Ben},
  title = {{Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX}},
  howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
  year = 2021,
  month = May
}
```

If you use this repository or any of the pretrained weights to do something cool, we would love to hear about it! Reach out on the [GitHub repo](https://github.com/kingoflolz/mesh-transformer-jax), Discord, or shoot Ben an email.

## Acknowledgements


This project would not have been possible without compute generously provided by the
[TPU Research Cloud](https://sites.research.google/trc/) and [EleutherAI](https://eleuther.ai/).

Thanks to the Cloud TPU team at Google for providing early access to the Cloud TPU VM alpha
([now publicly available!](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms))

Thanks to everyone who have helped out one way or another (listed alphabetically):
- [James Bradbury](https://twitter.com/jekbradbury) for valuable assistance with debugging JAX issues.
- [Stella Biderman](https://www.stellabiderman.com), Kurumuz, and [Finetune](https://github.com/finetuneanon/) for converting the model to be compatible with the `transformers` package.
- [Leo Gao](https://twitter.com/nabla_theta) for running zero shot evaluations for the baseline models for the table.
- [Laurence Golding](https://github.com/researcher2) for adding some features to the web demo.
- [Aran Komatsuzaki](https://twitter.com/arankomatsuzaki) for advice with experiment design and writing the blog posts.
- [Janko Prester](https://github.com/jprester) for creating the web demo frontend.