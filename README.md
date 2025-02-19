# VRAM Efficient Autograd

During LLM training, computing probabilities for every token takes a lot of VRAM. 

Using:

- Batch size: 4
- Sequence length: 4086
- Layer size: 4086
- Embedding size: 128K

Using blfloat16 requires 4GB of memory and 8GB in float32.

This autograd function reduces VRAM by **78%** vs. nn.CNE while preserving gradients.
