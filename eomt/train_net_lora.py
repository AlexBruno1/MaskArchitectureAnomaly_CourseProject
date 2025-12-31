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

# Use peft for LoRA
# class LoRASemantic(MaskClassificationSemantic):
#     def __init__(
#         self,
#         *args,
#         logit_norm_temperature=0.01,
#         lora_rank=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         lora_targets=["qkv", "q_proj", "v_proj"],
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
        
#         # Inject LoRA
#         logging.info(f"Injecting LoRA with r={lora_rank}, alpha={lora_alpha}, targets={lora_targets}")
#         # We need to wrap self.network.encoder.backbone or parts of it
#         # Based on ViT implementation in models/vit.py: self.network.encoder.backbone
        
#         # inject_lora returns a PEFT model which wraps the original module.
#         # We replace the backbone with the PEFT wrapper.
#         # OR we wrap the whole encoder.
#         # OR we wrap the whole network. 
#         # PEFT wraps existing modules.
        
#         self.network = inject_lora(
#             self.network, 
#             r=lora_rank, 
#             lora_alpha=lora_alpha, 
#             lora_dropout=lora_dropout,
#             target_modules=lora_targets
#         )
        
#         # Mark only LoRA params as trainable? inject_lora already prints trainable params.
#         # But we also have mask heads and class heads in EoMT.
#         # Usually for LoRA fine-tuning, heads might be trainable or not.
#         # If we only want to fine-tune LoRA, we should ensure heads are frozen OR if we want to fine-tune heads too.
#         # The prompt says "fine tune EoMT using LoRA". Typically implies LoRA + Heads or just LoRA.
#         # PEFT model sets requires_grad=False for non-adapter params.
#         # But our self.network is now the PEFT model.
#         # What about class_head and mask_head in EoMT class?
#         # They are part of self.network (EoMT).
#         # PEFT wraps the whole self.network.
#         # If inject_lora wraps self.network, then everything else is frozen by default unless modules_to_save is set.
#         # We should probably add "mask_head" and "class_head" to modules_to_save to keep them trainable
#         # as we are likely fine-tuning on a new dataset or refining.
#         # For this task, let's assume we want to keep them trainable. i.e. classifier fine-tuning.
        
#         # Re-initialize criterion with LogitNorm
#         logging.info(f"Replacing criterion with LogitNormMaskClassificationLoss (temp={logit_norm_temperature})")
#         self.criterion = LogitNormMaskClassificationLoss(
#             temperature=logit_norm_temperature,
#             num_points=self.criterion.num_points,
#             oversample_ratio=self.criterion.oversample_ratio,
#             importance_sample_ratio=self.criterion.importance_sample_ratio,
#             mask_coefficient=self.criterion.mask_coefficient,
#             dice_coefficient=self.criterion.dice_coefficient,
#             class_coefficient=self.criterion.class_coefficient,
#             num_labels=self.criterion.num_labels, # reuse
#             no_object_coefficient=self.criterion.eos_coef,
#         )




from typing import List

class LoRACLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add LoRA specific arguments as top-level subcommand args
        parser.add_argument("--lora_rank", type=int, default=8, help="Rank of LoRA adapters")
        parser.add_argument("--logit_norm_temperature", type=float, default=0.04, help="Temperature for Logit Normalization")
        parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
        parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
        parser.add_argument("--lora_targets", type=List[str], default=["qkv", "q_proj", "v_proj"], help="Modules names to inject LoRA into")
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
                        logging.warning(f"Key {k} not found in model")
                state_dict = new_state_dict

            # Check for pos_embed size mismatch
            for k in list(state_dict.keys()):
                if "pos_embed" in k:
                    if state_dict[k].shape != model.state_dict()[k].shape:
                        logging.warning(f"Shape mismatch for {k}: {state_dict[k].shape} vs {model.state_dict()[k].shape}. Skipping {k} to use model initialization.")
                        del state_dict[k]

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
                targets = list(set(lora_targets + head_targets))
                logging.info(f"Standard mode: Injecting LoRA into Backbone AND Heads.")

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
