

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
import copy
from torchvision import transforms 

from dataset import RIAWELCDataset, get_ria_welc_transforms
from utils import plot_confusion_matrix, calculate_metrics_report
from evaluate_tent import get_model_universal

def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Calculate the entropy of each sample"""
    return -(F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(dim=1)

def configure_model_for_rgda(model):

    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)) or 'LayerNorm' in m.__class__.__name__:
            m.train() 
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
            
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                
    return model

def evaluate_with_rgda(model, dataloader, device, class_names, args):

    
    student_model = configure_model_for_rgda(model)
    anchor_model = copy.deepcopy(student_model)
    anchor_model.eval()
    for param in anchor_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=args.tta_lr,
        betas=(0.9, 0.999)
    )
    
    print(f"\nEvaluating Pure RGDA (Model={args.model_name}, LR={args.tta_lr}, Margin={args.e_margin})...")
    
    all_labels, all_preds = [], []
    progress_bar = tqdm(dataloader, desc="RGDA-Execution")
    
    for inputs, labels in progress_bar:
        if inputs is None or labels is None: continue
        inputs, labels = inputs.to(device), labels.to(device)
        
        # --- Unified Forward Propagation and Entropy Filtering ---
        outputs = student_model(inputs)
        entropies = softmax_entropy(outputs)
        
        filter_ids = torch.where(entropies < args.e_margin)[0]
        
        if len(filter_ids) > 0:
            optimizer.zero_grad()
            selected_entropies = entropies[filter_ids]
            
            # Index weighting: Assigning higher weights to low-entropy samples
            weights = torch.exp(-selected_entropies)
            weights = weights / weights.sum()
            loss_ent = (weights * selected_entropies).sum()
            
            # Fisher Anchor Protection: Prevents the model from deviating significantly from the original structure
            loss_fisher = torch.tensor(0.0).to(device)
            if args.fisher_alpha > 0:
                for (name, param_s), (_, param_a) in zip(student_model.named_parameters(), anchor_model.named_parameters()):
                    if param_s.requires_grad:
                        loss_fisher += 0.5 * (param_s - param_a).pow(2).sum()
            
            loss_total = loss_ent + args.fisher_alpha * loss_fisher
            loss_total.backward()
            
            # Global anti-jitter cropping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_msg = f"{loss_total.item():.4f}"
        else:
            loss_msg = "StatsOnly"
            
        # --- Final Forecast Phase ---
        with torch.no_grad():
            student_model.eval()
            final_outputs = student_model(inputs)
            _, preds = torch.max(final_outputs, 1)
            
            # Reset BN to the 'train()' state to prepare for the next batch of statistics
            for m in student_model.modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)) or 'LayerNorm' in m.__class__.__name__:
                    m.train()
                    
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
        progress_bar.set_postfix({'Loss': loss_msg, 'Valid': f'{len(filter_ids)}/{len(inputs)}'})

    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    report_str = calculate_metrics_report(all_labels, all_preds, class_names)
    
    print(f"\nFinal Performance Result:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    
    return acc, f1, all_labels, all_preds, report_str

def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    class_to_idx = {'CR': 0, 'LP': 1, 'ND': 2, 'PO': 3}
    class_names = sorted(class_to_idx.keys(), key=class_to_idx.get)
    num_classes = len(class_names)
    
    test_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=False)
    
    test_loader = DataLoader(
        dataset=RIAWELCDataset(root_dir=args.test_data_dir, transform=test_transforms, class_to_idx=class_to_idx),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    model = get_model_universal(model_name=args.model_name, num_classes=num_classes, pretrained=False, checkpoint_path=args.model_path).to(device)
    
    test_acc, test_f1, test_labels, test_preds, report_str = evaluate_with_rgda(model, test_loader, device, class_names, args)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, f"{args.experiment_name}_report.txt")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(f"RGDA Pure Evaluation Results:\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Accuracy: {test_acc:.4f}\n")
            f.write(f"Macro F1: {test_f1:.4f}\n\n")
            f.write(report_str)
        
        cm_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_cm.png")
        plot_confusion_matrix(
            y_true=test_labels, 
            y_pred=test_preds, 
            class_names=class_names, 
            save_path=cm_save_path, 
            normalize=True
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output_rgda')
    parser.add_argument('--experiment_name', type=str, default='rgda_eval')
    
    parser.add_argument('--tta_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--e_margin', type=float, default=0.9)
    parser.add_argument('--fisher_alpha', type=float, default=0.0) 
    
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_gpu', action='store_true')
    
    args = parser.parse_args()
    if not any(arg.startswith('--use_gpu') for arg in os.sys.argv):
        args.use_gpu = True
    main(args)