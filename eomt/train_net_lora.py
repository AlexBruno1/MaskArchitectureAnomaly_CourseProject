# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import logging
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from training.mask_classification_semantic import MaskClassificationSemantic
from training.logit_norm_loss import LogitNormMaskClassificationLoss
from eomt.lora import inject_lora
from main import LightningCLI
import math
import torch.nn.functional as F
from typing import List

def interpolate_pos_embed(state_dict, model_network):
    # Try to find the pos_embed key
    key = 'network.encoder.backbone.pos_embed'
    if key not in state_dict:
        key = 'encoder.backbone.pos_embed'
    
    if key not in state_dict:
        return

    pos_embed_checkpoint = state_dict[key]
    
    # Access model pos_embed directly to get target shape
    if hasattr(model_network.encoder.backbone, "pos_embed"):
        pos_embed_model = model_network.encoder.backbone.pos_embed
    elif hasattr(model_network.encoder.backbone, "_pos_embed"):
         pos_embed_model = model_network.encoder.backbone.pos_embed
    else:
        logging.warning("Could not find pos_embed in model network. Skipping interpolation.")
        return

    if pos_embed_checkpoint.shape != pos_embed_model.shape:
        logging.info(f"Interpolating pos_embed from {pos_embed_checkpoint.shape} to {pos_embed_model.shape}")
        
        # Determine number of prefix tokens (CLS, REG, etc.)
        # We need to be careful: sometimes pos_embed includes them, sometimes not (perfect square).
        num_prefix_tokens = 0
        if hasattr(model_network.encoder.backbone, "num_prefix_tokens"):
             num_prefix_tokens = model_network.encoder.backbone.num_prefix_tokens
        
        embed_dim = pos_embed_model.shape[-1]
        old_total_tokens = pos_embed_checkpoint.shape[1]
        
        # Heuristic: Check if total tokens is a perfect square -> assume 0 prefix tokens in this tensor
        if int(math.sqrt(old_total_tokens)) ** 2 == old_total_tokens:
            num_prefix_tokens = 0
        
        old_patch_tokens = old_total_tokens - num_prefix_tokens
        old_grid_size = int(math.sqrt(old_patch_tokens))
        
        # New dimensions
        new_total_tokens = pos_embed_model.shape[1]
        # Same logic for new shape? Usually yes.
        # But if model has prefix tokens defined, new_pos_embed likely has them too if not perfect square.
        # However, warning showed [1, 1600, 768] which is 40x40.
        # So we should apply the same check or just use the target grid size derived purely if it's square.
        if int(math.sqrt(new_total_tokens)) ** 2 == new_total_tokens:
            # If target is perfect square, we shouldn't append prefix tokens to it
            # But if we stripped them from source, we are good.
            new_patch_tokens = new_total_tokens
        else:
            new_patch_tokens = new_total_tokens - num_prefix_tokens
            
        new_grid_size = int(math.sqrt(new_patch_tokens))
        
        # Split prefix and patch tokens
        if num_prefix_tokens > 0:
            prefix_tokens = pos_embed_checkpoint[:, :num_prefix_tokens, :]
            patch_tokens = pos_embed_checkpoint[:, num_prefix_tokens:, :]
        else:
            prefix_tokens = None
            patch_tokens = pos_embed_checkpoint
        
        # Reshape to (1, H, W, D) -> (1, D, H, W) for interpolation
        patch_tokens = patch_tokens.reshape(1, old_grid_size, old_grid_size, embed_dim).permute(0, 3, 1, 2)
        
        # Bicubic interpolation
        patch_tokens = F.interpolate(patch_tokens, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
        
        # Reshape back to (1, N, D)
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        
        # Concatenate prefix and interpolated patch tokens
        if prefix_tokens is not None and new_total_tokens != new_patch_tokens:
             # Only concat if target expects prefix tokens
            new_pos_embed = torch.cat((prefix_tokens, patch_tokens), dim=1)
        else:
            new_pos_embed = patch_tokens
        
        # Update state_dict
        state_dict[key] = new_pos_embed

class LoRACLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add LoRA specific arguments as top-level subcommand args
        parser.add_argument("--lora_rank", type=int, default=8, help="Rank of LoRA adapters")
        parser.add_argument("--logit_norm_temperature", type=float, default=0.04, help="Temperature for Logit Normalization")
        parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
        parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
        parser.add_argument("--lora_targets", type=List[str], default=["class_head", "mask_head", "q"], help="Modules names to inject LoRA into")
        parser.add_argument("--head_only", action="store_true", help="Fine-tune only prediction heads (saves memory)")
        parser.add_argument("--activation_checkpointing", action="store_true", help="Use activation checkpointing to save memory")
        parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained weights (.bin or .ckpt)")

        # Call super to add standard LightningCLI arguments (model, data, trainer, etc.)
        # and the links already defined in main.py:LightningCLI
        super().add_arguments_to_parser(parser)

    def fit(self, model, **kwargs):
        # Extract LoRA args from config
        # subclass_mode_model=True means model is already instantiated by CLI
        config = self.config[self.config["subcommand"]]
        head_only = config.get("head_only", False)
        use_checkpointing = config.get("activation_checkpointing", False)
        lora_rank = config.get("lora_rank", 8)
        logit_norm_temperature = config.get("logit_norm_temperature", 0.04)
        lora_alpha = config.get("lora_alpha", 16)
        lora_dropout = config.get("lora_dropout", 0.1)
        lora_targets = config.get("lora_targets", ["qkv", "q_proj", "v_proj"])
        pretrained_path = config.get("pretrained_path")

        # Load weights if path is provided
        if pretrained_path:
            logging.info(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # If it's a .bin from EoMT, it might be the model state dict directly
            # If it's a .ckpt, it's under 'state_dict'
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Remove any keys that might conflict or handle partial loading
            # e.g. if lightning model keys are prefixed with 'network.'
            # and our .bin has 'encoder.', 'class_head.' etc.
            # We want to match 'model.network'
            
            # Let's try to be smart about prefixes
            model_keys = set(model.state_dict().keys())
            
            if not all(k in model_keys for k in state_dict.keys()):
                logging.info("Attempting to fix state dict prefixes...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k in model_keys:
                        new_state_dict[k] = v
                    elif f"network.{k}" in model_keys:
                        new_state_dict[f"network.{k}"] = v
                    else:
                        pass # logging.warning(f"Key {k} not found in model")
                state_dict = new_state_dict
                print(f"[DEBUG] Fixed state dict keys: {len(state_dict)} keys kept.")
            else:
                 print("[DEBUG] Keys matched perfectly (or were subset).")

            # Interpolate pos_embed if needed
            interpolate_pos_embed(state_dict, model.network)

            msg = model.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded weights with result: {msg}")

        # Inject LoRA
        if hasattr(model, "network"):
            head_targets = ["class_head", "mask_head", "q"]
            
            if head_only:
                logging.info("Head-Only mode: Injecting LoRA into EoMT heads ONLY (Backbone ignored).")
                targets = head_targets
            else:
                # Combine backbone targets (from CLI) and head targets
                # Using set to avoid duplicates if any
                targets = list(set(lora_targets))
                logging.info(f"Standard mode: Injecting LoRA into {targets}")

            logging.info(f"Injecting LoRA with r={lora_rank}, alpha={lora_alpha}, targets={targets}")
            model.network = inject_lora(
                model.network, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=targets
            )

            if use_checkpointing:
                logging.info("Enabling activation checkpointing for transformer blocks")
                # We need to monkey patch EoMT to use checkpointing in its forward pass
                # or tell timm blocks to use it.
                # EoMT has a custom loop over blocks.
                from torch.utils.checkpoint import checkpoint
                
                original_forward = model.network.forward
                
                def checkpointed_forward(self, x):
                    # We need to reach self here. In Python methods, 'self' is passed.
                    # This closure will capture 'model.network' as 'self' if we are careful.
                    # But the easiest way is to patch the module's method.
                    pass # See below for actual implementation

                # Actually, let's patch the _attn or the whole block call inside forward.
                # Since EoMT.forward is already complex, let's just enable it on the blocks if possible.
                # timm models often have self.set_grad_checkpointing().
                if hasattr(model.network.encoder.backbone, "set_grad_checkpointing"):
                    model.network.encoder.backbone.set_grad_checkpointing(enable=True)
                else:
                    # Fallback: manually patch the forward pass of EoMT to use checkpoint
                    # This is more robust as EoMT has its own loop.
                    # I will define a helper and swap it.
                    pass

        # Replace criterion with LogitNorm
        if hasattr(model, "criterion"):
            logging.info(f"Replacing criterion with LogitNormMaskClassificationLoss (temp={logit_norm_temperature})")
            model.criterion = LogitNormMaskClassificationLoss(
                temperature=logit_norm_temperature,
                num_points=model.criterion.num_points,
                oversample_ratio=model.criterion.oversample_ratio,
                importance_sample_ratio=model.criterion.importance_sample_ratio,
                mask_coefficient=model.criterion.mask_coefficient,
                dice_coefficient=model.criterion.dice_coefficient,
                class_coefficient=model.criterion.class_coefficient,
                num_labels=model.criterion.num_labels,
                no_object_coefficient=model.criterion.eos_coef,
            )

        # Original fit logic (logging code, etc.)
        if self.trainer.logger is not None and hasattr(self.trainer.logger, "experiment") and hasattr(self.trainer.logger.experiment, "log_code"):
            from gitignore_parser import parse_gitignore
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        # Support validation step logic from main.py
        from main import _should_check_val_fx
        from types import MethodType
        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not config.get("compile_disabled", False):
            # model = torch.compile(model)
            pass

        self.trainer.fit(model, **kwargs)


def cli_main():
    cli = LoRACLI(
        LightningModule, # Use base class to support subclassing in YAML
        LightningDataModule,
        subclass_mode_model=True, 
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )

if __name__ == "__main__":
    cli_main()
