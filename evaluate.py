import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from safetensors.torch import load_file
import random
import numpy as np  
import timm  

from dataset import RIAWELCDataset, get_ria_welc_transforms
from model import get_convnextv2_model
from utils import plot_confusion_matrix, calculate_metrics_report

# ==================== General Model Loading Function ====================
def get_universal_model(model_name, num_classes, checkpoint_path=None, device='cpu'):

    print(f"Creating model architecture: {model_name}")
    try:

        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception as e:
        print(f"Error creating model '{model_name}': {e}")
        print("Tip: Ensure model_name is a valid timm architecture (e.g., 'resnet50', 'convnextv2_tiny')")
        raise e

    # Load local weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:

            if checkpoint_path.endswith('.safetensors'):
                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            

            state_dict = checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Detected checkpoint dictionary, extracting 'model_state_dict'...")
                state_dict = checkpoint['model_state_dict']
            

            if any(k.startswith('module.') for k in state_dict.keys()):
                print("Detected 'module.' prefix (DataParallel), removing it...")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}


            msg = model.load_state_dict(state_dict, strict=True)
            print(f"Weights loaded successfully. {msg}")
            
        except Exception as e:
            print(f"Error loading weights with strict=True: {e}")
            print("Retrying with strict=False (ignoring missing/mismatched keys)...")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Weights loaded with strict=False.")
            except Exception as e2:
                print(f"CRITICAL: Failed to load weights: {e2}")
                raise e2
    else:
        print(f"Warning: Model path '{checkpoint_path}' does not exist! Model uses random initialization.")
    
    return model.to(device)

def evaluate_model(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds =  []

    print("\nEvaluating on the test set...")
    
    try:
        from tqdm import tqdm
        iterator = tqdm(dataloader, desc="Testing")
    except ImportError:
        iterator = dataloader

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(iterator):
            if inputs is None or labels is None:
                continue
            

            if i == 0:
                print(f"\n[DEBUG] Input Batch Statistics (Batch {i}):")
                print(f"  Min: {inputs.min().item():.4f}")
                print(f"  Max: {inputs.max().item():.4f}")
                print(f"  Mean: {inputs.mean().item():.4f}")
                print(f"  Shape: {inputs.shape}")

                if inputs.max() > 50: 
                    print("  [WARNING] ⚠️ 警告：检测到原始像素值 (0-255)！")
                    print("  [SOLUTION] 请检查 transforms，必须包含 transforms.ToTensor() 且 Normalize 参数正确。")
                else:
                    print("  [INFO] 数据范围正常 (0-1 或标准化范围)。")
            # ============================================

            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    if not all_labels:
        print("No data processed.")
        return None, None, None, None, None, None

    test_loss = running_loss / len(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print(f"\n{'='*30}")
    print(f"Test Set Performance:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  macro F1-score: {f1:.4f}")
    print(f"{'='*30}\n")
    
    report_str = calculate_metrics_report(all_labels, all_preds, class_names)
    print(report_str)
    
    return test_loss, test_acc, f1, all_labels, all_preds, report_str

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

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")


    class_to_idx = {'CR': 0, 'LP': 1, 'ND': 2, 'PO': 3}
    class_names = sorted(class_to_idx.keys(), key=class_to_idx.get)
    num_classes = len(class_names)
    print(f"Class Mapping: {class_to_idx}")


    test_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=False)

    if not os.path.exists(args.test_data_dir):
        print(f"Error: Test data directory '{args.test_data_dir}' does not exist.")
        return
        
    test_dataset = RIAWELCDataset(
        root_dir=args.test_data_dir, 
        transform=test_transforms, 
        class_to_idx=class_to_idx
    )
    
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty. Check data path and content.")
        return

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    print(f"Test samples: {len(test_dataset)}")


    try:
        model = get_universal_model(
            model_name=args.model_name,
            num_classes=num_classes,
            checkpoint_path=args.model_path,
            device=device
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return


    print(f"Loading checkpoint from: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Detected checkpoint dictionary, extracting 'model_state_dict'...")
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if any(k.startswith('module.') for k in state_dict.keys()):
            print("Detected 'module.' prefix (DataParallel), removing it...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}


        msg = model.load_state_dict(state_dict, strict=True)
        print(f"Weights loaded successfully. {msg}")
        
    except Exception as e:
        print(f"CRITICAL ERROR loading weights: {e}")
        print("尝试设置 strict=False 继续加载...")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Weights loaded with strict=False (some layers might be missing).")
        except Exception as e2:
            print(f"Still failed: {e2}")
            return

    model.to(device)


    criterion = nn.CrossEntropyLoss()


    test_loss, test_acc, test_f1, test_labels, test_preds, report_str = evaluate_model(
        model, test_loader, criterion, device, class_names
    )


    if test_labels is not None:
        os.makedirs(args.output_dir, exist_ok=True)


        cm_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_cm.png")
        plot_confusion_matrix(test_labels, test_preds, class_names, save_path=cm_save_path)
        print(f"Confusion matrix saved to {cm_save_path}")


        report_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_report.txt")
        with open(report_save_path, "w", encoding='utf-8') as f:
            f.write(f"Test Set Performance for model: {args.model_path}\n")
            f.write(f"Class Mapping Used: {class_to_idx}\n")
            f.write(f"  Loss: {test_loss:.4f}\n")
            f.write(f"  Accuracy: {test_acc:.4f}\n")
            f.write(f"  Weighted F1-score: {test_f1:.4f}\n\n")
            f.write(report_str)
        print(f"Report saved to {report_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ConvNeXt V2 for X-ray Weld Defect Classification')
    

    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model file')
    

    parser.add_argument('--output_dir', type=str, default='./output_evaluation', help='Dir to save results')
    parser.add_argument('--experiment_name', type=str, default='eval_result', help='Prefix for saved files')
    parser.add_argument('--model_name', type=str, default='convnextv2_tiny.fcmae_ft_in1k', help='Model architecture name')
    

    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Drop path rate (0 for eval usually)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    

    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu', help='Force CPU usage')

    args = parser.parse_args()
    main(args)