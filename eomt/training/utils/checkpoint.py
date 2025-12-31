
import logging

def fix_state_dict_keys(state_dict, model):
    """
    Fixes state_dict keys to match the model's keys.
    Specifically handles the case where the checkpoint is from a raw backbone (e.g. timm)
    and the model is the EoMT wrapper which prefixes the backbone.
    """
    
    # First pass: clean prefixes and identify potential matches
    model_keys = set(model.state_dict().keys())
    
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if "base_model.model." in new_key:
            new_key = new_key.replace("base_model.model.", "")
        
        # Determine if this is a LoRA key or base key
        # possible suffixes: .base_layer.weight, .lora_A.default.weight, .lora_B.default.weight
        if ".base_layer." in new_key:
            clean_stem = new_key.replace(".base_layer.weight", "").replace(".base_layer.bias", "")
            cleaned_state_dict[new_key] = (clean_stem, v)
        elif ".lora_" in new_key:
            # e.g. network.q.lora_A.default.weight -> network.q
            if "lora_A.default.weight" in new_key:
                clean_stem = new_key.replace(".lora_A.default.weight", "")
                cleaned_state_dict[new_key] = (clean_stem, v)
            elif "lora_B.default.weight" in new_key:
                clean_stem = new_key.replace(".lora_B.default.weight", "")
                cleaned_state_dict[new_key] = (clean_stem, v)
            else:
                 # Other lora keys (dropout etc)
                cleaned_state_dict[new_key] = (None, v)
        else:
            cleaned_state_dict[new_key] = (new_key, v)

    # Group by potential parameter name (e.g. network.q.weight)
    # We want to form target keys like 'network.q.weight'
    # Check candidates
    
    # We need to reconstruct the state dict
    final_dict = {}
    
    # Analyze stems
    # Map from stem -> { 'base': tensor, 'lora_A': tensor, 'lora_B': tensor }
    extracts = {}
    
    pass_through_keys = {}
    
    for k, (stem, v) in cleaned_state_dict.items():
        if stem is None: 
            continue # Skip irrelevant lora keys
            
        # Check if this stem maps to a real weight in model
        # Target: stem + ".weight" or stem + ".bias" usually
        # But wait, stem could be "network.q"
        # We expect "network.q.weight" in model
        
        if k.endswith(".base_layer.weight"):
            extracts.setdefault(stem, {})['base_weight'] = v
        elif k.endswith(".base_layer.bias"):
            extracts.setdefault(stem, {})['base_bias'] = v
        elif k.endswith(".lora_A.default.weight"):
            extracts.setdefault(stem, {})['lora_A'] = v
        elif k.endswith(".lora_B.default.weight"):
            extracts.setdefault(stem, {})['lora_B'] = v
        else:
            # Regular key
            pass_through_keys[k] = v

    # Process extracts
    lora_alpha = 16 # Default from train_net_lora.py
    for stem, parts in extracts.items():
        # Reconstruct "weight"
        if 'base_weight' in parts:
            base = parts['base_weight']
            if 'lora_A' in parts and 'lora_B' in parts:
                A = parts['lora_A']
                B = parts['lora_B']
                rank = A.shape[0]
                scaling = lora_alpha / rank
                
                # Merge: W = base + B @ A * scaling
                # A: (r, in), B: (out, r) -> B @ A: (out, in)
                # Check shapes
                if base.shape == (B @ A).shape:
                     merged_weight = base + (B @ A) * scaling
                     logging.info(f"Merged LoRA weights for {stem}.weight (rank={rank})")
                     final_dict[f"{stem}.weight"] = merged_weight
                else:
                    logging.warning(f"Shape mismatch for LoRA merge at {stem}: base {base.shape}, delta {(B@A).shape}. Using base.")
                    final_dict[f"{stem}.weight"] = base
            else:
                final_dict[f"{stem}.weight"] = base
        
        if 'base_bias' in parts:
            final_dict[f"{stem}.bias"] = parts['base_bias']
            
    # Process pass-throughs
    for k, v in pass_through_keys.items():
        final_dict[k] = v

    # Now filter against model keys to be safe and handle the original prefix logic
    # The final_dict now has keys like "network.class_head.weight" (stripped of base_model)
    # But we still need to run the flexible mapping loop just in case (e.g. valid checks)
    
    new_state_dict = {}
    mapped_count = 0
    skipped_count = 0
    
    for k, v in final_dict.items():
        # Try direct match first (most likely now)
        if k in model_keys:
            new_state_dict[k] = v
            mapped_count += 1
            continue
            
        # Try prefixes (legacy logic)
        candidates = [
            f"network.{k}",
            f"network.encoder.{k}",
            f"network.encoder.backbone.{k}"
        ]
        
        found = False
        for cand in candidates:
            if cand in model_keys:
                new_state_dict[cand] = v
                mapped_count += 1
                found = True
                break
        
        if not found:
             skipped_count += 1

    logging.info(f"Remapped {mapped_count} keys, skipped {skipped_count} keys.")
    return new_state_dict
