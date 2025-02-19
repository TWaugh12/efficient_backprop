# efficient_backprop

Large Language Models (LLMs) compute next-token probabilities using a projection matrix, represented as:

$$
\Large \sigma(XW)
$$

where:

- $\Large X$ is the input embedding matrix,  
- $\Large W$ is the weight matrix mapping to a large vocabulary,  
- $\Large \sigma$ represents softmax activation.  

For a vocabulary of size 128k, computing logits at full precision causes VRAM spikes and costly memory usage. For example, with:

- Batch size: $4$
- Sequence length: $4096$
- Hidden dimension: $4096$
- Vocabulary size: $128K$

The logit matrix in blfloat16 requires **4GB** of memory, and **8GB** if upcast to float32.

The custom autograd function reduces peak VRAM while preserving gradients using the chain rule:

$$
\Large
\frac{dL}{dW} = X_{1}^T  \frac{dL_1}{dy_1} + X_2^T  \frac{dL_2}{dy_2}
$$

This method successfully reduces memory requirements by **78%** while maintaining numerical stability (evaluated against nn.CNE).
