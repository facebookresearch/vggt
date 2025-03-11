import torch
import argparse
from pathlib import Path
import numpy as np

def load_checkpoint(checkpoint_path):
    """Load a PyTorch checkpoint file."""
    return torch.load(checkpoint_path, map_location=torch.device('cpu'))

def compare_checkpoints(reference_ckpt, other_ckpts):
    """
    Compare a reference checkpoint with multiple other checkpoints.
    
    Args:
        reference_ckpt: The reference checkpoint (A)
        other_ckpts: List of other checkpoints to compare against (B, C, D, etc.)
        
    Returns:
        Dictionary with comparison results for each checkpoint
    """
    ref_state_dict = reference_ckpt
    if isinstance(ref_state_dict, dict) and 'state_dict' in ref_state_dict:
        ref_state_dict = ref_state_dict['state_dict']
    
    results = {}
    
    for ckpt_name, ckpt in other_ckpts.items():
        other_state_dict = ckpt
        if isinstance(other_state_dict, dict) and 'state_dict' in other_state_dict:
            other_state_dict = other_state_dict['state_dict']
            
        # Initialize result categories
        results[ckpt_name] = {
            'matching_keys_and_values': [],
            'matching_keys_different_values': [],
            'keys_only_in_reference': [],
            'keys_only_in_other': []
        }
        
        # Recursively compare dictionaries
        compare_dicts_recursive(ref_state_dict, other_state_dict, '', results[ckpt_name])
    
    return results

def compare_dicts_recursive(ref_dict, other_dict, prefix, results):
    """
    Recursively compare two dictionaries and update results.
    
    Args:
        ref_dict: Reference dictionary
        other_dict: Other dictionary to compare against
        prefix: Current key prefix for nested dictionaries
        results: Dictionary to store results
    """
    # Check keys in reference that exist in other
    for key in ref_dict.keys():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if key in other_dict:
            ref_val = ref_dict[key]
            other_val = other_dict[key]
            
            # If both values are dictionaries, recurse
            if isinstance(ref_val, dict) and isinstance(other_val, dict):
                compare_dicts_recursive(ref_val, other_val, full_key, results)
            else:
                # Handle different types of values
                if isinstance(ref_val, torch.Tensor) and isinstance(other_val, torch.Tensor):
                    # Both are tensors
                    if ref_val.shape == other_val.shape:
                        if torch.allclose(ref_val, other_val, rtol=1e-5, atol=1e-5):
                            results['matching_keys_and_values'].append(full_key)
                        else:
                            results['matching_keys_different_values'].append(full_key)
                    else:
                        results['matching_keys_different_values'].append(full_key)
                elif isinstance(ref_val, (int, float)) and isinstance(other_val, (int, float)):
                    # Both are numbers
                    if abs(ref_val - other_val) < 1e-5:
                        results['matching_keys_and_values'].append(full_key)
                    else:
                        results['matching_keys_different_values'].append(full_key)
                elif ref_val == other_val:
                    # Other types (strings, bools, etc.)
                    results['matching_keys_and_values'].append(full_key)
                else:
                    results['matching_keys_different_values'].append(full_key)
        else:
            results['keys_only_in_reference'].append(full_key)
    
    # Find keys only in other dictionary
    for key in other_dict.keys():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in ref_dict:
            results['keys_only_in_other'].append(full_key)

def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch checkpoints')
    parser.add_argument('--reference', '-r', type=str, required=True, 
                        help='Path to reference checkpoint A')
    parser.add_argument('--others', '-o', type=str, nargs='+', required=True,
                        help='Paths to other checkpoints (B, C, D, etc.)')
    args = parser.parse_args()
    
    # Load reference checkpoint
    print(f"Loading reference checkpoint: {args.reference}")
    reference_ckpt = load_checkpoint(args.reference)
    
    # Load other checkpoints
    other_ckpts = {}
    for path in args.others:
        name = Path(path).stem
        print(f"Loading checkpoint: {path}")
        other_ckpts[name] = load_checkpoint(path)
    
    # Compare checkpoints
    results = compare_checkpoints(reference_ckpt, other_ckpts)
    
    # Print results
    print("\n===== COMPARISON RESULTS =====")
    for ckpt_name, result in results.items():
        print(f"\n## Comparing reference with {ckpt_name}:")
        
        print(f"\n  Keys with matching names but DIFFERENT values ({len(result['matching_keys_different_values'])}):")
        for key in sorted(result['matching_keys_different_values']):
            print(f"  - {key}")
            
        print(f"\n  Keys with matching names and values ({len(result['matching_keys_and_values'])}):")
        print(f"  - {len(result['matching_keys_and_values'])} keys match exactly")
        
        print(f"\n  Keys only in reference ({len(result['keys_only_in_reference'])}):")
        print(f"  - {len(result['keys_only_in_reference'])} keys only in reference")
        
        print(f"\n  Keys only in {ckpt_name} ({len(result['keys_only_in_other'])}):")
        print(f"  - {len(result['keys_only_in_other'])} keys only in {ckpt_name}")

if __name__ == "__main__":
    main()
