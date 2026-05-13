

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

def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    return -(probs * F.log_softmax(logits, dim=1)).sum(dim=1)

def configure_model_for_sar(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    return model

class SAMOptimizer:

    def __init__(self, params, lr, rho=0.05):
        self.params = list(params)
        self.lr = lr
        self.rho = rho
        self.optimizer = optim.SGD(self.params, lr=lr)
        self.state = {} # Used to store the state corresponding to the parameters (such as disturbance terms)

    @torch.no_grad()
    def first_step(self):
        """Finding the perturbation direction epsilon"""
        grad_norm = self._grad_norm()
        for p in self.params:
            if p.grad is None: continue
            eps = p.grad * (self.rho / (grad_norm + 1e-12))
            p.add_(eps)  
            self.state[p] = eps # Store the perturbation term in the class's internal state dictionary

    @torch.no_grad()
    def second_step(self):
        """Return to the starting point and update using the gradient at the perturbation point"""
        for p in self.params:
            if p.grad is None: continue
            if p in self.state:
                p.sub_(self.state[p])  
        

        torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in self.params if p.grad is not None]), 
            2
        )
        return norm

def evaluate_with_sar(model, dataloader, device, class_names, args):
    """使用完整版 SAR 算法进行评估"""
    model = configure_model_for_sar(model)

    sam = SAMOptimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=args.sar_lr, rho=0.05)
    
    all_labels = []
    all_preds = []
    total_selected = 0
    total_samples = 0
    
    print(f"\nEvaluating with Real SAR (SAM-Optimized, LR={args.sar_lr})...")
    progress_bar = tqdm(dataloader, desc="SAR-Full")
    
    for inputs, labels in progress_bar:
        if inputs is None or labels is None:
            continue
        
        inputs, labels = inputs.to(device), labels.to(device)
        total_samples += inputs.size(0)
        
        # --- SAM Step 1: Finding the Right Focus ---
        logits = model(inputs)
        entropy = compute_entropy(logits)
        

        max_conf = logits.max(dim=1)[0]
        selected_ids = torch.where(max_conf > args.denoise_margin)[0]
        total_selected += len(selected_ids)
        
        if len(selected_ids) > 0:
            loss = entropy[selected_ids].mean()
            loss.backward()
            sam.first_step() 
            
            # --- SAM Step 2: Compute the gradient at the domain point and update ---
            sam.optimizer.zero_grad()
            logits_adv = model(inputs)
            entropy_adv = compute_entropy(logits_adv)
            loss_adv = entropy_adv[selected_ids].mean()
            loss_adv.backward()
            sam.second_step() 
            
            loss_val = loss.item()
        else:
            loss_val = 0.0
            

        with torch.no_grad():
            model.eval()
            final_logits = model(inputs)
            _, preds = torch.max(final_logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            model.train()
        
        progress_bar.set_postfix({'loss': f'{loss_val:.4f}', 'sel': f'{len(selected_ids)}'})
    

    selection_ratio = total_selected / total_samples if total_samples > 0 else 0
    test_acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    report_str = calculate_metrics_report(all_labels, all_preds, class_names)
    
    print(f"\nSelection Ratio: {selection_ratio:.2%}")
    print(f"Final Accuracy: {test_acc:.4f}, F1: {f1:.4f}")
    
    return test_acc, f1, all_labels, all_preds, report_str

def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    
    class_to_idx = {'CR': 0, 'LP': 1, 'ND': 2, 'PO': 3}
    class_names = sorted(class_to_idx.keys(), key=class_to_idx.get)
    
    test_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=False)
    test_loader = DataLoader(
        RIAWELCDataset(root_dir=args.test_data_dir, transform=test_transforms, class_to_idx=class_to_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    model = get_model_universal(
        model_name=args.model_name, num_classes=len(class_names), 
        pretrained=False, checkpoint_path=args.model_path
    ).to(device)
    
    acc, f1, labels, preds, report = evaluate_with_sar(model, test_loader, device, class_names, args)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, f"{args.experiment_name}_report.txt")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(f"SAR Full Algorithm Results (No Cheating)\nAcc: {acc:.4f}\nF1: {f1:.4f}\n\n{report}")
        
        cm_path = os.path.join(args.output_dir, f"{args.experiment_name}_cm.png")
        plot_confusion_matrix(labels, preds, class_names, save_path=cm_path, normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output_sar')
    parser.add_argument('--experiment_name', type=str, default='sar_full')
    parser.add_argument('--sar_lr', type=float, default=1e-3)
    parser.add_argument('--denoise_margin', type=float, default=0.5) 
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_gpu', action='store_true')
    
    args = parser.parse_args()
    if not any(arg.startswith('--use_gpu') for arg in os.sys.argv): args.use_gpu = True
    main(args)