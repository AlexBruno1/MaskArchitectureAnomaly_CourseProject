# eomt/diagnose_vit.py
import inspect
import sys
import os

# Force import from local folder
sys.path.append(os.getcwd())

try:
    from models.vit import ViT
    print("SUCCESS: Found ViT class.")
except ImportError as e:
    print(f"ERROR: Could not import ViT. {e}")
    sys.exit(1)

print("\n" + "="*50)
print("REQUIRED ARGUMENTS FOR ViT:")
print("="*50)

# Print the exact arguments the model wants
sig = inspect.signature(ViT.__init__)
print(str(sig))

print("\n" + "="*50)
print("ACTION REQUIRED:")
print("Copy the line starting with '(self, ...)' above and paste it in the chat.")
print("="*50)