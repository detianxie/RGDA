

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import timm
from safetensors.torch import load_file

from dataset import RIAWELCDataset, get_ria_welc_transforms
from utils import plot_confusion_matrix, calculate_metrics_report


# ==================== 通用模型加载函数 ====================

def get_model_universal(model_name, num_classes=4, pretrained=False, checkpoint_path=None):

    # 模型名称映射表 - 支持简写和完整名称
    MODEL_NAME_MAPPING = {
        # ConvNeXtV2
        'convnextv2': 'convnextv2_tiny.fcmae_ft_in1k',
        'convnextv2_tiny': 'convnextv2_tiny.fcmae_ft_in1k',
        'convnextv2_base': 'convnextv2_base.fcmae_ft_in1k',
        
        # EfficientNetV2
        'efficientnetv2': 'efficientnetv2_s.in1k',
        'efficientnetv2_s': 'efficientnetv2_s',
        'efficientnetv2_m': 'efficientnetv2_m.in1k',
        
        # MobileNetV3
        'mobilenetv3': 'mobilenetv3_large_100.ra_in1k',
        'mobilenetv3_large': 'mobilenetv3_large_100.ra_in1k',
        
        # RepVGG
        'repvgg': 'repvgg_b0.rvgg_in1k',
        'repvgg_a0': 'repvgg_a0',
        'repvgg_b1': 'repvgg_b1.rvgg_in1k',
        
        # ResNet50
        'resnet50': 'resnet50.a1_in1k',
        'resnet50_a1': 'resnet50.a1_in1k',
        
        # Swin Transformer
        'swin': 'swin_tiny_patch4_window7_224.ms_in1k',
        'swin_tiny': 'swin_tiny_patch4_window7_224.ms_in1k',
        'swin_base': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
        
        # ViT
        'vit': 'vit_base_patch16_224.augreg_in1k',
        'vit_base': 'vit_base_patch16_224.augreg_in1k',
        'vit_small': 'vit_small_patch16_224.augreg_in1k',
    }
    

    model_key = model_name.lower().replace('-', '_')
    timm_model_name = MODEL_NAME_MAPPING.get(model_key, model_name)
    
    print(f"Loading model: {timm_model_name}")
    

    try:
        model = timm.create_model(
            timm_model_name,
            pretrained=pretrained if checkpoint_path is None else False,
            num_classes=num_classes
        )
    except Exception as e:
        print(f"Error creating model '{timm_model_name}': {e}")
        print(f"Available models containing '{model_name}': ")
        available = timm.list_models(f"*{model_name}*")
        for m in available[:10]:
            print(f"  - {m}")
        raise
    

    if checkpoint_path:
        print(f"Loading pre-trained weights from a local path: {checkpoint_path}")
        

        if checkpoint_path.endswith(".safetensors"):
            checkpoint_state_dict = load_file(checkpoint_path)
        elif checkpoint_path.endswith(".pth"):
            checkpoint_state_dict = torch.load(checkpoint_path, map_location='cpu')

            if 'model_state_dict' in checkpoint_state_dict:
                print("Checkpoint dictionary format detected; extracting... 'model_state_dict'...")
                checkpoint_state_dict = checkpoint_state_dict['model_state_dict']
            # ======================================
        else:
            raise ValueError("This format is not supported. Please use .pth or .safetensors")
        

        model_state_dict = model.state_dict()
        model_state_dict = model.state_dict()
        new_state_dict = {
            k: v for k, v in checkpoint_state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        

        model.load_state_dict(new_state_dict, strict=False)
        

        loaded_keys = new_state_dict.keys()
        total_keys = checkpoint_state_dict.keys()
        print(f"Loaded successfully {len(loaded_keys)} / {len(total_keys)} matching weight layer")
        
        if len(loaded_keys) != len(total_keys):
            ignored_keys = [k for k in total_keys if k not in loaded_keys]
            print(f"Number of ignored layers: {len(ignored_keys)}")
            if len(ignored_keys) <= 10:
                print(f"The ignored layer (usually the classification header): {ignored_keys}")
    
    return model


# ==================== Tent TTACore Logic ====================

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Calculate the entropy of the softmax distribution for the logits"""
    return -(F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(dim=1).mean()


def configure_model_for_tent(model, model_name=''):

    model.eval()
    

    for param in model.parameters():
        param.requires_grad = False
    

    model_name_lower = model_name.lower()
    

    if any(x in model_name_lower for x in ['convnext', 'swin', 'vit', 'transformer']):
        norm_layer = nn.LayerNorm
        print("Using LayerNorm for Tent adaptation (Transformer-based model)")
    else:

        norm_layer = nn.BatchNorm2d
        print("Using BatchNorm2d for Tent adaptation (CNN-based model)")
    

    for m in model.modules():
        if isinstance(m, norm_layer):

            m.train() 

            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False 


            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    
    return model


def test_time_adapt_tent(model, dataloader, device, learning_rate=1e-3):

    # Create an optimizer only for trainable parameters (the affine parameters of the normalization layer)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)
    
    all_labels = []
    all_preds = []
    
    print(f"\nEvaluating on the test set with Tent (Test-Time Adaptation)...")
    print(f"Total trainable parameters: {sum(p.numel() for p in params)}")
    
    progress_bar = tqdm(dataloader, desc="TTA Evaluation")
    
    for inputs, labels in progress_bar:
        if inputs is None or labels is None:
            continue
        
        inputs, labels = inputs.to(device), labels.to(device)
        

        optimizer.zero_grad()
        

        outputs = model(inputs)
        

        loss = softmax_entropy(outputs)
        

        loss.backward()
        optimizer.step()
        

        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        

        progress_bar.set_postfix(entropy_loss=f'{loss.item():.4f}')
    
    return all_labels, all_preds


# ==================== Main function ====================

def main(args):



    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_everything(42)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    class_to_idx = {'CR': 0, 'LP': 1, 'ND': 2, 'PO': 3}

    class_names = sorted(class_to_idx.keys(), key=class_to_idx.get)
    num_classes = len(class_names)
    

    test_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=False)
    test_dataset = RIAWELCDataset(
        root_dir=args.test_data_dir, 
        transform=test_transforms, 
        class_to_idx=class_to_idx
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    print(f"\nTest samples: {len(test_dataset)}")
    

    model = get_model_universal(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=False,
        checkpoint_path=args.model_path
    ).to(device)
    

    model = configure_model_for_tent(model, model_name=args.model_name)
    

    all_labels, all_preds = test_time_adapt_tent(
        model, 
        test_loader, 
        device, 
        learning_rate=args.tent_lr
    )
    

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print(f"\n{'='*50}")
    print(f"Classification Report:\n")
    print(calculate_metrics_report(all_labels, all_preds, class_names))
    
    print(f"\nTest Set Performance with Tent:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  macro Avg Precision: {precision:.4f}")
    print(f"  macro Avg Recall: {recall:.4f}")
    print(f"  macro Avg F1-score: {f1:.4f}")
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    cm_path = os.path.join(args.output_dir, f"{args.experiment_name}_cm.png")
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    report_path = os.path.join(args.output_dir, f"{args.experiment_name}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Checkpoint: {args.model_path}\n")
        f.write(f"Test Dataset: {args.test_data_dir}\n")
        f.write(f"Tent Learning Rate: {args.tent_lr}\n")
        f.write(f"\n{calculate_metrics_report(all_labels, all_preds, class_names)}\n")
        f.write(f"\nTest Set Performance with Tent:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  macro Avg F1-score: {f1:.4f}\n")
    
    print(f"TTA evaluation report saved to {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal Tent TTA Evaluation for All Models')
    

    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (支持: convnextv2, efficientnetv2, mobilenetv3, repvgg, resnet50, swin, vit)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth or .safetensors)')
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help='Path to test dataset (e.g., SWXD/test, RIAWELC/test, GD_Xray/test)')
    

    parser.add_argument('--output_dir', type=str, default='./tent_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--experiment_name', type=str, default='tent_eval',
                        help='Experiment name for output files')
    
   
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    
    parser.add_argument('--tent_lr', type=float, default=1e-3,
                        help='Learning rate for Tent adaptation')
    
    args = parser.parse_args()
    main(args)