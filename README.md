# Swin-Transformers-from-scratch

![image](https://user-images.githubusercontent.com/60067496/203227984-a126de0d-2d70-48e5-8fe8-46e707b4ff34.png)

shifted window approach for computing self-attention in the proposed Swin Transformer architecture. In layer l (left), a regular window partitioning scheme is
adopted, and self-attention is computed within each window. In the next layer l + 1 (right), the window partitioning is shifted, resulting in new windows. The self-attention computation in the new
windows crosses the boundaries of the previous windows in layer l, providing connections among them.

![image](https://user-images.githubusercontent.com/60067496/203228183-fa031b29-668c-4676-869f-498a79bcb5a3.png)

 (a) The architecture of a Swin Transformer (Swin-T); (b) two successive Swin Transformer Blocks. W-MSA and SW-MSA are multi-head self attention modules with regular and shifted windowing configurations, respectively
 
Swin Transformer is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on
shifted windows, with other layers kept the same. A Swin Transformer block consists of a shifted window based MSA
module, followed by a 2-layer MLP with GELU nonlinearity in between. A LayerNorm (LN) layer is applied
before each MSA module and each MLP, and a residual connection is applied after each module

### Shifted Windows
The standard Transformer architecture and its adaptation for image classification both conduct global selfattention, where the relationships between a token and all
other tokens are computed. The global computation leads to
quadratic complexity with respect to the number of tokens,
making it unsuitable for many vision problems requiring an
immense set of tokens for dense prediction or to represent a
high-resolution image.
