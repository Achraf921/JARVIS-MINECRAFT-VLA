# Notes:

(A couple of notes from parts of the paper I deemed "important" to understand I've decided to cache here before building)

## Training

Pre-training : (Traditional VLA training) : Regular next-token Imitation learning with all components hot
Post-training :

ActVLP paradigm:


    Step 1 : World knowledge post-training, frozen ViT, hot LLM, training on regular world knowledge text question with image context tokens from Vit


    Step 2 : Visual knowledge (based on the Vit's embeddings) questions with both Vit and llm hot
             visual question-answering (VQA) with the following training objective
<img src="./views/Step2Objective.png" width="300"> 

(regular GPT next token cross entropy)

     Step 3 : Action post-training, VLM now predicts action tokens which influences the action sequence loop, we train it to predict an action which is a sequence of action tokens At:t+tau in a specific order to product the right action/action sequence, ViT remains frozen

<img src="./views/step3_nll_post_tr.png" width="300">

(regular nll over the token sequence)


## Model Structure :

As displayed here :

<img src="./views/architecture.png" width="600">

With one note, the Causal Transformer is not the cross-attention kind, where we run cross attention on the llm's Q embeddings using the Vit's KV cache to compute attention, here, it simply scales the Vit's output embeddings to the llm's dimensionality and appends them at the end of the prompt output (using boundary tokens apparently e.g : <|vision_start|>, <|vision_end|>)
and then runs regular masked self-attention

Architecture :

```

Image input -> ViT -> Image projection module (simple 2-layer MLP)-------|
                                                                         |
                                                        Regular Autoregressive llm (GPT) ------> Action Decoder --> Action token (discrete) sequence
                                                                         |
Prompt ---- -------------------------------------------------------------|

```

Prompt : 

The prompt include the text prompt as well as HISTORY of several previous Images (well their Vit + MLP output tokens) that are crucial for longer horizon tasks/multi-step reasoning

Proposed Base models : Llava-Next or Qwen2-VL

(Since this paper is a year old, we'll try to use better/more recent versions of those and see where it takes us)

Tokenization :

Instead of re-training the base VLM's tokenizer, we just repurpose the 51 last frequently used tokens into action tokens with the following split :
- 22 mouse control tokens
- 29 keyboard input tokens

! No other modification to the initial VLM's architecture is made which allows us to implement jarvis with a variety of foundational models


## Dataset :

This illustration is highkey enough : 

<img src="./views/dataset.png" width="700">
