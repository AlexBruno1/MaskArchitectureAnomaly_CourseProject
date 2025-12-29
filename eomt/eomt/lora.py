import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

def inject_lora(model, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "q_proj", "v_proj", "query", "value"]):
    """
    Injects LoRA adapters into the model using PEFT.
    
    Args:
        model: The PyTorch model to adapt.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to target. 
                        Common ViT/Transformer targets: "qkv", "q_proj", "v_proj".
                        Timm models often use "qkv", transformers uses "query", "value".
    """
    
    # Filter target modules that actually exist in the model
    # beneficial to avoid warnings or errors if some modules are not present
    existing_modules = []
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name.split('.')[-1]:
                 existing_modules.append(target)
    
    unique_targets = list(set(existing_modules))
    
    if not unique_targets:
        # Fallback to standard ViT targets if auto-detection fails or strict list provided
        unique_targets = target_modules

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=unique_targets,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=[], # Modules to unfreeze other than LoRA. e.g. "classifier"
    )
    
    # PEFT wraps the model.
    # Note: get_peft_model might wrap the model in a PeftModel class.
    # If the original code expects specific attributes on 'model', this might need handling.
    # However, PeftModel usually forwards attribute access.
    peft_model = get_peft_model(model, config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    return peft_model
