import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RIAWELCDataset(Dataset):
    """
    Custom Dataset for RIAWELC X-ray weld defect images.
    Assumes images are organized in subfolders named by class
    (e.g., root_dir/LP/image1.png, root_dir/PO/image2.png, etc.)
    """
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        """
        Args:
            root_dir (string): Directory with all the images, organized by class.
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict, optional): A mapping from class name to class index.
                                           If None, it will be inferred from subdirectories.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # Image storage path
        self.labels = []      # Store the corresponding label

        if class_to_idx is None:
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(list(self.class_to_idx.keys()))

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory {class_dir} not found.")
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')): # RIAWELC uses.png [1]
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # RIAWELC images are 8-bit PNGs [1], likely grayscale.
            # ConvNeXt V2 expects 3-channel RGB input. [2, 3]
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image and label or raise an error
            # For simplicity, returning None here, handle appropriately in DataLoader or training loop
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_ria_welc_transforms(image_size=224, is_train=True):
    """
    Returns a composition of transforms for the RIAWELC dataset.
    Uses ImageNet mean and std for normalization as ConvNeXt V2 models are often pretrained on it. [4]
    """
    # ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        # Data augmentation for training [5, 3, 6, 7]
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),      # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.5),        # Random vertical flip
            transforms.RandomRotation(degrees=15),       # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color dithering
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    else:
        # Transforms for validation/testing (no random augmentation)
        transform_list = [
            transforms.Resize((image_size, image_size)),
        
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    return transforms.Compose(transform_list)

def create_data_loaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=4, image_size=224):

    # Defining the category mapping for the RIAWELC dataset
    class_to_idx = {'LP': 0, 'PO': 1, 'CR': 2, 'ND': 3}
    class_names = ['LP', 'PO', 'CR', 'ND']
    num_classes = len(class_names)
    
    # Data Retrieval and Transformation
    train_transforms = get_ria_welc_transforms(image_size=image_size, is_train=True)
    val_transforms = get_ria_welc_transforms(image_size=image_size, is_train=False)
    
    # Create a dataset
    datasets = {}
    loaders = {}
    
    # Training dataset
    if train_dir and os.path.exists(train_dir):
        train_dataset = RIAWELCDataset(
            root_dir=train_dir, 
            transform=train_transforms, 
            class_to_idx=class_to_idx
        )
        datasets['train'] = train_dataset
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        print(f"The training set has been loaded: {len(train_dataset)} image")
    else:
        print(f"Warning: Training data directory {train_dir} Does not exist")
        loaders['train'] = None
    
    # Validation dataset
    if val_dir and os.path.exists(val_dir):
        val_dataset = RIAWELCDataset(
            root_dir=val_dir, 
            transform=val_transforms, 
            class_to_idx=class_to_idx
        )
        datasets['val'] = val_dataset
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"The validation set has finished loading: {len(val_dataset)} image")
    else:
        print(f"Warning: Validation data directory {val_dir} Does not exist")
        loaders['val'] = None
    
    # Test dataset
    if test_dir and os.path.exists(test_dir):
        test_dataset = RIAWELCDataset(
            root_dir=test_dir, 
            transform=val_transforms, 
            class_to_idx=class_to_idx
        )
        datasets['test'] = test_dataset
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"The test set has been loaded.: {len(test_dataset)} image")
    else:
        print(f"Warning: Test data directory {test_dir} Does not exist")
        loaders['test'] = None
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    for split, dataset in datasets.items():
        print(f"{split.upper()} Dataset:")
        print(f"  Total number of images: {len(dataset)}")
        
        # Count the number of images in each category
        class_counts = {class_name: 0 for class_name in class_names}
        for label in dataset.labels:
            class_name = class_names[label]
            class_counts[class_name] += 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} image")
        print()
    
    return loaders['train'], loaders['val'], loaders['test'], class_names, num_classes

def test_data_loading(data_loader, dataset_name="Dataset"):

    if data_loader is None:
        print(f"{dataset_name} The data loader is empty; skipping the test")
        return
    
    print(f"\n=== test {dataset_name} Data Loader ===")
    try:
        # Retrieve a batch of data
        images, labels = next(iter(data_loader))
        
        print(f"Batch size: {images.shape[0]}")
        print(f"Image Shape: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"Image Data Types: {images.dtype}")
        print(f"Tag Data Types: {labels.dtype}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Tag value: {labels.tolist()}")
        
        # Check for None values
        if images.isnan().any():
            print("Warning: NaN values detected in the image data")
        
        print(f"{dataset_name} Data loading test successful！")
        
    except Exception as e:
        print(f"test {dataset_name} An error occurred while loading the data: {e}")

if __name__ == '__main__':
    # Data Path Configuration - Please modify this according to your actual path
    DATA_ROOT = "D:\shujvji\lunwen\data"  # Master Data Catalog
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")      # Training data
    VAL_DIR = os.path.join(DATA_ROOT, "val")          # Validate data  
    TEST_DIR = os.path.join(DATA_ROOT, "test")        # Test data
    

    BATCH_SIZE = 16
    NUM_WORKERS = 4
    IMAGE_SIZE = 224
    
    print("=== RIAWELC Dataset Loader Testing ===")
    
    # Create a data loader
    train_loader, val_loader, test_loader, class_names, num_classes = create_data_loaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR, 
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )
    
    print(f"\nCategory Name: {class_names}")
    print(f"Number of categories: {num_classes}")
    

    test_data_loading(train_loader, "training set")
    test_data_loading(val_loader, "Validation set") 
    test_data_loading(test_loader, "Test set")
    
   