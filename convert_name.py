import os
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_checkpoint_parameter_names(
    checkpoint_path, 
    output_path=None,
    name_mappings=None,
    load_to_model=None,
    strict=False
):
    """
    Load weights from a checkpoint file and convert parameter names
    based on provided mappings.
    
    Args:
        checkpoint_path (str): Path to the .pt checkpoint file
        output_path (str, optional): Path to save the converted checkpoint.
                                    If None, doesn't save to disk.
        name_mappings (dict, optional): Dictionary of {old_prefix: new_prefix} mappings.
                                       Default: {'image_backbone': 'patch_embed'}
        load_to_model (nn.Module, optional): If provided, loads weights to this model
        strict (bool): Whether to strictly enforce that the keys in state_dict match
                      the keys returned by this module's state_dict()
    
    Returns:
        dict: The converted state dictionary
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None
    
    # Default mapping if none provided
    if name_mappings is None:
        name_mappings = {'image_backbone': 'patch_embed'}
    
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Create a new state dict with converted keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            
            # Apply all mappings
            for old_pattern, new_pattern in name_mappings.items():
                if old_pattern in key:
                    new_key = key.replace(old_pattern, new_pattern)
                    logger.debug(f"Converted: {key} -> {new_key}")
                    break
            
            new_state_dict[new_key] = value
        
        # Save the converted checkpoint if output path is provided
        if output_path:
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                checkpoint['model'] = new_state_dict
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint['state_dict'] = new_state_dict
            else:
                checkpoint = new_state_dict
                
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            torch.save(checkpoint, output_path)
            logger.info(f"Saved converted checkpoint to {output_path}")
        
        # Load to model if provided
        if load_to_model is not None:
            missing_keys, unexpected_keys = load_to_model.load_state_dict(new_state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.info(f"Successfully loaded weights to model with parameter name conversion")
        
        return new_state_dict
        
    except Exception as e:
        logger.error(f"Error processing checkpoint: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    # You can add more mappings to this dictionary as needed
    mappings = {
        'image_backbone': 'patch_embed',
        '.pose_token': '.camera_token',
        # 'pose_branch': 'camera_branch',
        # Add more mappings here in the future, for example:
        # 'old_attention': 'new_attention',
        # 'old_mlp': 'new_mlp',
    }
    
    # Convert checkpoint without loading to a model
    convert_checkpoint_parameter_names(
        checkpoint_path='/fsx-repligen/jianyuan/cvpr2025_ckpts/r518_t7_cmh_v7_0-d4w770q_model.pt',
        output_path='/fsx-repligen/jianyuan/cvpr2025_ckpts/r518_t7_cmh_v7_0-d4w770q_model_converted.pt',
        name_mappings=mappings
    )
    
    # Example with loading to a model:
    """
    import your_model_module
    model = your_model_module.YourModel()
    convert_checkpoint_parameter_names(
        checkpoint_path='path/to/input.pt',
        name_mappings=mappings,
        load_to_model=model,
        strict=False
    )
    """





