

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
    return -(F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(dim=1).mean()

def configure_model_for_mid_range(model: nn.Module) -> nn.Module:

    model.train() 
    
    # Freeze all parameters except BN
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze BN parameters
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
            
            # CoTTA Standard Operating Procedures
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                
    return model

@torch.no_grad()
def update_teacher(teacher_model, student_model, alpha):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

@torch.no_grad()
def stochastic_restore(student_model, anchor_model, restore_prob):

    for student_param, anchor_param in zip(student_model.parameters(), anchor_model.parameters()):
        if not student_param.requires_grad:
            continue
        mask = (torch.rand_like(student_param) < restore_prob).float()
        student_param.data.copy_(mask * anchor_param.data + (1 - mask) * student_param.data)

def evaluate_cotta(model, dataloader, device, class_names, args):
    model = configure_model_for_mid_range(model)
    
    teacher_model = copy.deepcopy(model)
    for param in teacher_model.parameters(): param.requires_grad = False
        
    anchor_model = copy.deepcopy(model)
    for param in anchor_model.parameters(): param.requires_grad = False
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.tta_lr,
        betas=(0.9, 0.999)
    )
    
    tta_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.72, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ])

    print(f"\nEvaluating CoTTA Mid-Range Target (Restore p={args.restore_prob}, LR={args.tta_lr})...")
    
    all_labels = []
    all_preds =  []

    progress_bar = tqdm(dataloader, desc="CoTTA-Mid")
    for inputs, labels in progress_bar:
        if inputs is None or labels is None:
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 1. Teacher
        with torch.no_grad():
            outputs_teacher = teacher_model(inputs)
            preds_teacher = outputs_teacher.argmax(dim=1)
        
        # 2. Student
        outputs = model(inputs)
        loss_ent = softmax_entropy(outputs)
        
        # 3. Augmentation
        try:
            inputs_aug = tta_transform(inputs)
        except:
            inputs_aug = inputs
        outputs_aug = model(inputs_aug)
        loss_cons = F.cross_entropy(outputs_aug, preds_teacher)
        
        loss = loss_ent + args.lambda_cons * loss_cons
        loss.backward()
        optimizer.step()
        
        update_teacher(teacher_model, model, alpha=args.alpha)
        
        
        stochastic_restore(model, anchor_model, restore_prob=args.restore_prob)
        
        with torch.no_grad():
            final_outputs = teacher_model(inputs)
            _, preds = torch.max(final_outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    test_acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    report_str = calculate_metrics_report(all_labels, all_preds, class_names)
    
    print(f"\nMid-Range Result -> Acc: {test_acc:.4f}, F1: {f1:.4f}")
    return test_acc, f1, all_labels, all_preds, report_str

def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    
    class_to_idx = {'CR': 0, 'LP': 1, 'ND': 2, 'PO': 3}
    class_names = sorted(class_to_idx.keys(), key=class_to_idx.get)
    num_classes = len(class_names)
    
    test_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=False)
    test_loader = DataLoader(
        RIAWELCDataset(root_dir=args.test_data_dir, transform=test_transforms, class_to_idx=class_to_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    model = get_model_universal(
        model_name=args.model_name,
        num_classes=num_classes,
        checkpoint_path=args.model_path
    ).to(device)

    test_acc, test_f1, test_labels, test_preds, report_str = evaluate_cotta(
        model, test_loader, device, class_names, args
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, f"{args.experiment_name}_report.txt")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(f"CoTTA Mid-Range Results:\n")
            f.write(f"Acc: {test_acc:.4f}\nF1: {test_f1:.4f}\n\n")
            f.write(report_str)
        print(f"Saved to {report_path}")
# ================= Add code to save the confusion matrix =================
        if test_labels is not None and len(test_labels) > 0:
            cm_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_cm.png")
            plot_confusion_matrix(
                y_true=test_labels, 
                y_pred=test_preds, 
                class_names=class_names, 
                save_path=cm_save_path, 
                normalize=True  
            )
            print(f"Confusion matrix saved to {cm_save_path}")
        # ==========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output_cotta_mid')
    parser.add_argument('--experiment_name', type=str, default='cotta_mid')
    
    # === Key parameter combinations ===

    parser.add_argument('--tta_lr', type=float, default=1e-3)
    
    parser.add_argument('--alpha', type=float, default=0.999)
    

    parser.add_argument('--restore_prob', type=float, default=0.5)
    
    parser.add_argument('--lambda_cons', type=float, default=1.0)
    
    parser.add_argument('--model_name', type=str, default='convnextv2_tiny.fcmae_ft_in1k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_gpu', action='store_true')
    
    args = parser.parse_args()
    if not any(arg.startswith('--use_gpu') for arg in os.sys.argv):
        args.use_gpu = True
    main(args)