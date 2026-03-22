import torch
from transformers import AutoModelForCausalLM, AutoConfig
import argparse

def transmute_granite(source_path: str, target_path: str):
    print(f"Summoning the beast from {source_path}...")
    
    # 1. Load the configuration
    config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)
    
    # Untie the weights to sever the bond between embedding and LM head
    was_tied = getattr(config, "tie_word_embeddings", False)
    if was_tied:
        print("Severing the tied weights between embed_tokens and lm_head...")
        config.tie_word_embeddings = False

    # 2. Load the model directly into CPU RAM
    model = AutoModelForCausalLM.from_pretrained(
        source_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Extract the runes of power
    m_e = getattr(config, "embedding_multiplier", 1.0)
    m_r = getattr(config, "residual_multiplier", 1.0)
    m_l = getattr(config, "logits_scaling", 1.0)

    print(f"Extracted Multipliers -> Embedding: {m_e}, Residual: {m_r}, Logits: {m_l}")

    # 3. Transmute the Embedding
    print("Baking embedding multiplier into embed_tokens...")
    model.model.embed_tokens.weight.data.mul_(m_e)

    # 4. Transmute the Residuals (Upgraded to handle biases)
    print("Baking residual multiplier into attention and MLP output projections...")
    for i, layer in enumerate(model.model.layers):
        # Scale the Attention output projection
        layer.self_attn.o_proj.weight.data.mul_(m_r)
        if layer.self_attn.o_proj.bias is not None:
            layer.self_attn.o_proj.bias.data.mul_(m_r)

        # Scale the SwiGLU MLP output projection
        layer.mlp.down_proj.weight.data.mul_(m_r)
        if layer.mlp.down_proj.bias is not None:
            layer.mlp.down_proj.bias.data.mul_(m_r)

    # 5. Transmute the Logits
    print("Baking logits scaling into the LM head...")
    if was_tied or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
        model.lm_head.weight.data.div_(m_e * m_l)
    else:
        model.lm_head.weight.data.div_(m_l)

    # 6. Cleanse the configuration
    config.embedding_multiplier = 1.0
    config.residual_multiplier = 1.0
    config.logits_scaling = 1.0

    # 7. Seal the new artifact
    print(f"Binding the transmuted titan to {target_path}...")
    model.save_pretrained(target_path)
    config.save_pretrained(target_path)
    print("Transmutation complete. The beast is ready to cross the Bifröst.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake Granite multipliers into HF weights for Megatron compatibility.")
    parser.add_argument("--source", type=str, required=True, help="Path to original Hugging Face model")
    parser.add_argument("--target", type=str, required=True, help="Path to save the Megatron-ready HF model")
    
    args = parser.parse_args()
    transmute_granite(args.source, args.target)