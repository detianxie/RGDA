import torch
import timm
import torch.nn as nn
from safetensors.torch import load_file

def get_convnextv2_model(model_name='convnextv2_tiny.fcmae_ft_in1k',
                         num_classes=4,
                         pretrained=True,
                         drop_path_rate=0.1,
                         checkpoint_path=None):


    model = timm.create_model(
        model_name,
        pretrained=pretrained if checkpoint_path is None else False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )


    if checkpoint_path:
        print(f"Loading pre-trained weights from a local path: {checkpoint_path}")
        

        if checkpoint_path.endswith(".safetensors"):
            checkpoint_state_dict = load_file(checkpoint_path)
        elif checkpoint_path.endswith(".pth"):

            checkpoint_state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            raise ValueError("This format is not supported. Please use .pth or .safetensors")


        model_state_dict = model.state_dict()
        new_state_dict = {
            k: v for k, v in checkpoint_state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }


        model.load_state_dict(new_state_dict, strict=False)
        

        loaded_keys = new_state_dict.keys()
        total_keys = checkpoint_state_dict.keys()
        print(f"成功加载 {len(loaded_keys)} / {len(total_keys)} 个匹配的权重层。")
        if len(loaded_keys) != len(total_keys):
            print(f"被忽略的层 (通常是分类头): {[k for k in total_keys if k not in loaded_keys]}")
    
    return model





if __name__ == '__main__':
    # Example usage:
    print("Testing ConvNeXt V2 model loading...")
    # Example: Load ConvNeXtV2 Tiny pretrained on ImageNet, with a new head for 4 classes
    model = get_convnextv2_model(model_name='convnextv2_tiny.fcmae_ft_in1k', num_classes=4, pretrained=True)
    print(f"Model: {model.__class__.__name__}")
    # print(model) # Uncomment to see model structure

    # Test with a dummy input
    import torch
    dummy_input = torch.randn(2, 3, 224, 224) # Batch size 2, 3 channels, 224x224 image
    try:
        output = model(dummy_input)
        print(f"Dummy input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}") # Should be (batch_size, num_classes)
    except Exception as e:
        print(f"Error during model forward pass: {e}")

    # Example: Load ConvNeXtV2 Base
    model_base = get_convnextv2_model(model_name='convnextv2_base.fcmae_ft_in1k', num_classes=4, pretrained=True, drop_path_rate=0.1) # Drop path for base is often higher, e.g., 0.1-0.2 [5]
    print(f"Model: {model_base.__class__.__name__}")