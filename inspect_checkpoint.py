import torch
import sys

def inspect_checkpoint(checkpoint_path):
    """
    Loads a checkpoint and inspects its weights for numerical instability (nan/inf).
    """
    if not checkpoint_path:
        print("Please provide a path to the checkpoint file.")
        return

    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        # Load the checkpoint onto the CPU to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # The model's weights are usually in a dictionary named 'model_state_dict' or 'model'
        model_state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")

        if model_state_dict is None:
            print("ERROR: Could not find 'model_state_dict' or 'model' in the checkpoint.")
            print("Available keys:", checkpoint.keys())
            return

        print("Inspecting model weights...")
        found_issue = False
        
        for name, param in model_state_dict.items():
            if torch.isnan(param).any():
                print(f"!!! Found NaN in layer: {name}")
                found_issue = True
            if torch.isinf(param).any():
                print(f"!!! Found Inf in layer: {name}")
                found_issue = True
        
        if not found_issue:
            print("\n✅ All weights in the checkpoint appear to be valid (no NaNs or Infs found).")
        else:
            print("\n❌ Found numerical issues in the checkpoint. The file is likely corrupted.")

    except Exception as e:
        print(f"An error occurred while inspecting the checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_file = sys.argv[1]
        inspect_checkpoint(checkpoint_file)
    else:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint_file>")
