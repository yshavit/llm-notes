(beyond-the-toy-llm)=

# Beyond the toy LLM

:::{warning} WIP
TODO

- GELU for activation function

- RoPE

  This replaces the positional embeddings we discussed last time. I need to learn this before I can write it up. :-)

  e.g. [RoPE (Medium)](https://medium.com/@mlshark/rope-a-detailed-guide-to-rotary-position-embedding-in-modern-llms-fde71785f152)

- In the real world, there are multiple self-attention-plus-NN layers:

  ```mermaid
  flowchart LR
    Text[Input text] --> Embeddings
    subgraph Layer1[Transformer layer 1]
      S1[Self-attention] --> N1[FFN]
    end
    Embeddings --> S1[Transformer layer 2]
    subgraph Layer2
      S2[Self-attention] --> N2[FFN]
    end
    N1 --> S2
    subgraph LayerN[More transformer layers...]
      direction LR
      SN[Self-attention] --> NN[FFN]
    end
    N2 --> LayerN
    LayerN --> Output
  ```

- what are the real world numbers here?
  - how many transformer layers
  - what are the various dimensionalities

- layer normalization has two learned params (`https://claude.ai/chat/629d0e9e-8517-40c8-a9a2-518533ce25b4`), beta and gamma
:::
