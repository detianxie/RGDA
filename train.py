import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import timm  


from dataset import RIAWELCDataset, get_ria_welc_transforms
from utils import plot_learning_curves, plot_confusion_matrix, calculate_metrics_report

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num, num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch_num+1}/{num_epochs}", leave=False)

    for inputs, labels in progress_bar:
        if inputs is None or labels is None:
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        

        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # ====================================================

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    

    return epoch_loss, epoch_acc.item()

def validate_one_epoch(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    if not all_labels:
        return 0, 0, 0, [], []

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"  Val Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {f1:.4f}")
    
    return epoch_loss, epoch_acc, f1, all_labels, all_preds

def get_universal_model(model_name, num_classes, pretrained=True, checkpoint_path=None):

    print(f"Creating model: {model_name}")
    try:
        
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:
        print(f"Error creating model '{model_name}': {e}")
        
        simple_name = model_name.split('.')[0]
        if simple_name != model_name:
            print(f"Retrying with simplified name: {simple_name}")
            model = timm.create_model(simple_name, pretrained=pretrained, num_classes=num_classes)
        else:
            raise e


    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint for fine-tuning: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Warning loading checkpoint: {e}")

    return model

def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")


    class_to_idx = {'CR': 0, 'LP': 1, 'ND': 2, 'PO': 3} 
    class_names = sorted(class_to_idx.keys(), key=class_to_idx.get)
    num_classes = len(class_names)

    train_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=True)
    val_transforms = get_ria_welc_transforms(image_size=args.image_size, is_train=False)

    train_dataset_path = os.path.join(args.data_dir, 'train')
    val_dataset_path = os.path.join(args.data_dir, 'val')


    if not os.path.exists(train_dataset_path):
        print(f"Error: Path not found: {train_dataset_path}")
        return

    train_dataset = RIAWELCDataset(root_dir=train_dataset_path, transform=train_transforms, class_to_idx=class_to_idx)
    val_dataset = RIAWELCDataset(root_dir=val_dataset_path, transform=val_transforms, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")


    model = get_universal_model(
             model_name=args.model_name,
             num_classes=num_classes,

            pretrained=(not args.checkpoint_path) and (not args.scratch),
             checkpoint_path=args.checkpoint_path
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min if args.lr_min else args.lr * 0.01)

    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    print(f"Start training {args.model_name} for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_acc, val_f1, val_labels, val_preds = validate_one_epoch(model, val_loader, criterion, device, class_names)
        
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)


        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(args.output_dir, f"{args.experiment_name}_best_model.pth")
            

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_acc': val_acc
            }, best_model_path)
            
            print(f"  [Save] New best model saved! (F1: {best_val_f1:.4f})")
            

            plot_confusion_matrix(val_labels, val_preds, class_names, 
                                  save_path=os.path.join(args.output_dir, f"{args.experiment_name}_best_cm.png"))
            with open(os.path.join(args.output_dir, f"{args.experiment_name}_report.txt"), "w") as f:
                f.write(calculate_metrics_report(val_labels, val_preds, class_names))

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes.")
    print(f"Best Validation F1-score: {best_val_f1:.4f}")
    

    plot_learning_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'], 
                         save_path=os.path.join(args.output_dir, f"{args.experiment_name}_curves.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ConvNeXt V2 for X-ray Weld Defect Classification')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of the dataset (containing train/val subfolders)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save models and logs')
    parser.add_argument('--experiment_name', type=str, default='convnextv2_ria_welc', help='Name for this experiment')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a local model checkpoint (.pth) to start training from (for fine-tuning).')
    
    parser.add_argument('--model_name', type=str, default='convnextv2_tiny.fcmae_ft_in1k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_min', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Number of epochs to train only the classifier head (0 to disable)')
    parser.add_argument('--lr_head', type=float, default=None, help='Learning rate for training the classifier head (e.g., 1e-3 or 2*lr)')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_gpu', action='store_true')

    parser.add_argument('--scratch', action='store_true', help='Train from scratch (no pre-training)')

    args = parser.parse_args()
    
    if not any(arg.startswith('--use_gpu') or arg.startswith('--no-use_gpu') for arg in os.sys.argv):
        args.use_gpu = True

    main(args)

