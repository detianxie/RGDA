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

def seed_everything(seed=42):
    """Ensure reproducibility."""
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
    """Compute the entropy of softmax distribution."""
    return -(F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(dim=1)

def configure_model_for_adaptation(model):
    """
    Configure model for TTA: update only the affine parameters of normalization layers.
    """
    model.train() 
    for param in model.parameters():
        param.requires_grad = False
    
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
            
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                
    return model

def evaluate_with_eata(model, dataloader, device, class_names, args):
    """Evaluate using the Efficient Test-Time Adaptation (EATA) algorithm."""
    student_model = configure_model_for_adaptation(copy.deepcopy(model))
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=args.tta_lr
    )
    
    print(f"\nEvaluating with EATA (LR={args.tta_lr}, Entropy Margin={args.e_margin})...")
    
    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc="EATA Evaluation")
    for inputs, labels in progress_bar:
        if inputs is None or labels is None:
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        
        # --- EATA Adaptation Step ---
        outputs = student_model(inputs)
        entropies = softmax_entropy(outputs)
        
        # EATA Core: Filter out high-entropy (unreliable) samples
        filter_ids = torch.where(entropies < args.e_margin)[0]
        
        loss = torch.tensor(0.0).to(device)
        if len(filter_ids) > 0:
            loss = entropies[filter_ids].mean()

        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Final Prediction Step ---
        with torch.no_grad():
            student_model.eval()
            final_outputs = student_model(inputs)
            _, preds = torch.max(final_outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        student_model.train()

    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    report_str = calculate_metrics_report(all_labels, all_preds, class_names)
    
    print(f"\nTest Set Performance with EATA:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1-score: {f1:.4f}")
    
    return test_acc, f1, all_labels, all_preds, report_str


def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    class_to_idx = {'CR': 0, 'LP': 1, 'ND': 2, 'PO': 3}
    class_names = sorted(class_to_idx.keys(), key=class_to_idx.get)
    num_classes = len(class_names)

    test_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=False)
    test_dataset = RIAWELCDataset(root_dir=args.test_data_dir, transform=test_transforms, class_to_idx=class_to_idx)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Test samples: {len(test_dataset)}")

    model = get_model_universal(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=False,
        checkpoint_path=args.model_path
    ).to(device)

    test_acc, test_f1, test_labels, test_preds, report_str = evaluate_with_eata(model, test_loader, device, class_names, args)

    if test_labels:
        os.makedirs(args.output_dir, exist_ok=True)
        cm_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_cm.png")
        plot_confusion_matrix(test_labels, test_preds, class_names, save_path=cm_save_path, normalize=True)
        
        report_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_report.txt")
        with open(report_save_path, "w", encoding='utf-8') as f:
            f.write(f"Test Set Performance with EATA (Model: {args.model_name}, LR: {args.tta_lr})\n")
            f.write(f"  Accuracy: {test_acc:.4f}\n")
            f.write(f"  Macro F1-score: {test_f1:.4f}\n\n")
            f.write(report_str)
        print(f"EATA evaluation report saved to {report_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model with EATA (Efficient TTA)')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to target domain data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to source pre-trained model')
    parser.add_argument('--output_dir', type=str, default='./output_eata')
    parser.add_argument('--experiment_name', type=str, default='eata_eval')
    
    parser.add_argument('--tta_lr', type=float, default=1e-3, help='Learning rate for EATA') 
    parser.add_argument('--e_margin', type=float, default=0.9, help='Entropy threshold for filtering')
    
    parser.add_argument('--model_name', type=str, default='convnextv2_tiny.fcmae_ft_in1k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()
    if not any(arg.startswith('--use_gpu') for arg in os.sys.argv):
        args.use_gpu = True
    main(args)