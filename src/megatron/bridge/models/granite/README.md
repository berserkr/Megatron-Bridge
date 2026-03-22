# Preparing Granite for Megatron-NeMo: The Transmutation Step

## The Problem: Granite's Custom Math vs. Megatron's Rigid Forge

The IBM Granite architecture introduces custom scalar multipliers into its forward pass to control signal variance and loss scaling. Specifically, it uses:

* `embedding_multiplier` ($m_e$): Scales the initial token embeddings.
* `residual_multiplier` ($m_r$): Scales the output of the Attention and SwiGLU MLP layers before adding them back to the residual stream.
* `logits_scaling` ($m_l$): Divides the final logits before cross-entropy loss computation.

Megatron Core is forged for raw, unyielding speed across massive GPU clusters. Its C++ and CUDA kernels (such as `get_bias_dropout_add`) are hardcoded for standard Transformer mathematics. They do not natively accept custom scalar multipliers in the residual stream or the embedding layer.

## The Dilemma

To run a Granite model in Megatron-NeMo, we must choose one of two paths:

1. **The Blacksmith's Way (Code Patching):** Fork Megatron Core, rewrite the core `TransformerLayer` and custom CUDA kernels to accept Granite's multipliers, and maintain this brittle fork forever.
2. **The Alchemist's Way (Weight Transmutation):** Leverage linear algebra to bake the scalar multipliers directly into the Hugging Face weight matrices *before* converting the model to Megatron format.

This repository chooses the Alchemist's Way.

## The Solution: Mathematical Baking

Because linear projections are commutative with scalar multiplication, multiplying the output of a layer by a scalar is mathematically identical to multiplying the layer's weights by that scalar.

However, we must account for the distributive property when biases are present. In Granite, the residual multiplier scales the entire output of the attention layer:


$$H_{out} = H_{in} + ((X W_o^T + b_o) \cdot m_r)$$

Distributed, this becomes:


$$H_{out} = H_{in} + X(W_o^T \cdot m_r) + (b_o \cdot m_r)$$

The `transmute_granite.py` script performs this one-time operation on the Hugging Face checkpoint:

* **Embeddings:** Multiplies `embed_tokens.weight` by $m_e$.
* **Residuals:** Multiplies the weights *and* biases of `o_proj` (Attention output) and `down_proj` (MLP output) by $m_r$.
* **Logits:** Divides `lm_head.weight` by $m_l$.

### The Tied-Weight Paradox

If the model ties its embedding and language model head weights (`tie_word_embeddings: true`), applying different scalars ($m_e$ for embeddings, $m_l$ for logits) to the same physical memory is impossible. The script automatically detects this, unties the weights, clones the tensor, applies the independent scalings, and saves them as distinct parameters.

## The Outcome

By running this script, we produce a "Megatron-ready" Hugging Face checkpoint. This checkpoint will yield the exact same mathematical outputs as the original Granite model, but it can now be converted via `AutoBridge` and run on unmodified, highly optimized Megatron-NeMo infrastructure.

## Usage

Run the transmutation script on your base Hugging Face model before invoking the Megatron conversion tools:

```bash
python transmute_granite.py \
    --source /path/to/original/granite-230b \
    --target /path/to/transmuted/granite-230b

```

Once complete, point the `AutoBridge` tools at the `--target` directory.
