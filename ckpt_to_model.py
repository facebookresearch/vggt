import torch
import os
import argparse

def extract_model_from_checkpoint(checkpoint_path):
    """
    Load a PyTorch checkpoint, extract the 'model' component, and save it as a new file.
    
    Args:
        checkpoint_path (str): Path to the original checkpoint file
    
    Returns:
        str: Path to the saved model file
    """
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract the model
    if "model" not in checkpoint:
        raise KeyError("The checkpoint does not contain a 'model' key")
    
    model_state_dict = checkpoint["model"]
    
    # Create the new filename
    base_name, ext = os.path.splitext(checkpoint_path)
    model_path = f"{base_name}_model{ext}"
    
    # Save the model
    print(f"Saving model to {model_path}")
    torch.save(model_state_dict, model_path)
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Extract model from checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()
    
    try:
        model_path = extract_model_from_checkpoint(args.checkpoint_path)
        print(f"Successfully extracted model to {model_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


# rsync -avz --progress --human-readable jianyuan@devvm29915.cln0:/home/jianyuan/tmp_data/ckpts/v0_model.pt .