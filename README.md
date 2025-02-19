# efficient_backprop
Large Language Models (LLMs) compute next-token probabilities using a projection matrix, represented as:
\[ \sigma(XW) \]
where:
- \( X \) is the input embedding matrix,
- \( W \) is the weight matrix mapping to a large vocabulary,
- \( \sigma \) represents softmax activation.

For a vocabulary of size 128K, computing logits at full precision causes VRAM spikes and inefficient memory usage. For example, with:
- Batch size: \( 4 \)
- Sequence length: \( 4096 \)
- Hidden dimension: \( 4096 \)
- Vocabulary size: \( 128K \)

the logit matrix in bfloat16 require **4GB** of memory, and **8GB** if upcast to float32.



The custom autograd function reduces peak VRAM while preserving gradients using the chain rule.

 \[
\begin{aligned}
\frac{dL}{dX} &= \begin{bmatrix} \frac{dL_1}{dy_1} W^T \\ \frac{dL_2}{dy_2} W^T \end{bmatrix} \\
\frac{dL}{dW} &= X_1^T \frac{dL_1}{dy_1} + X_2^T \frac{dL_2}{dy_2}
\end{aligned}
\]

This method reduces memory requirements by **78%** while maintaining numerical stability.
