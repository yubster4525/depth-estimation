import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class KittiDatasetAccessor:
    """Helper class to access KITTI dataset images.
    
    This class provides a simple way to iterate through images in a KITTI
    dataset split, following the standard KITTI file format.
    """
    
    def __init__(self, kitti_path, split="eigen", mode="test"):
        """
        Initialize KittiDatasetAccessor.
        
        Args:
            kitti_path (str): Path to KITTI raw dataset
            split (str): Split name ('eigen', 'eigen_zhou', or 'benchmark')
            mode (str): Mode ('train', 'val', or 'test')
        """
        self.kitti_path = kitti_path
        self.split = split
        self.mode = mode
        self.split_files = self._load_split_files()
        
    def _load_split_files(self):
        """Load split file list."""
        split_file_path = os.path.join(self.kitti_path, 'splits', self.split, f'{self.mode}_files.txt')
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file not found: {split_file_path}")
        
        with open(split_file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def __len__(self):
        """Return number of images in the split."""
        return len(self.split_files)
    
    def __iter__(self):
        """Iterate over images in the split."""
        for line in self.split_files:
            parts = line.split()
            
            # Handle different file formats
            if len(parts) >= 3:
                # Format: sequence frame_id camera
                seq = parts[0]
                stem = int(parts[1])
                side = parts[2]  # 'l' or 'r'
                
                # Map side to camera
                cam = "image_02" if side == "l" else "image_03"
                
                # Construct image path
                image_path = os.path.join(self.kitti_path, seq, cam, 'data', f'{stem:010d}.png')
            else:
                # Alternative format: direct path
                image_path = os.path.join(self.kitti_path, parts[0])
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Yield image and path identifier
            yield np.array(image), os.path.splitext(os.path.basename(image_path))[0]
    
    def get_image_path(self, idx):
        """Get path for image at index."""
        if idx >= len(self.split_files):
            raise IndexError(f"Index {idx} out of range for split with {len(self.split_files)} images")
            
        line = self.split_files[idx]
        parts = line.split()
        
        if len(parts) >= 3:
            # Format: sequence frame_id camera
            seq = parts[0]
            stem = int(parts[1])
            side = parts[2]  # 'l' or 'r'
            
            # Map side to camera
            cam = "image_02" if side == "l" else "image_03"
            
            # Construct image path
            return os.path.join(self.kitti_path, seq, cam, 'data', f'{stem:010d}.png')
        else:
            # Alternative format: direct path
            return os.path.join(self.kitti_path, parts[0])


class KittiDataset(Dataset):
    """PyTorch dataset for KITTI depth estimation.
    
    This dataset is designed for training depth estimation models
    on the KITTI dataset with ground truth from various sources.
    """
    
    def __init__(self, dataset_path, split='eigen', mode='train', transform=None, target_transform=None):
        """
        Initialize KittiDataset.
        
        Args:
            dataset_path (str): Path to KITTI dataset
            split (str): Split name ('eigen', 'eigen_zhou', or 'benchmark')
            mode (str): Mode ('train', 'val', or 'test')
            transform (callable, optional): Transform for input images
            target_transform (callable, optional): Transform for depth maps
        """
        self.dataset_path = dataset_path
        self.split = split
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        
        # Load split files
        split_file = os.path.join(dataset_path, 'splits', split, f'{mode}_files.txt')
        if not os.path.exists(split_file):
            raise ValueError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        
        print(f"Found {len(self.file_list)} samples in {split}/{mode} split")
    
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Get item by index."""
        # Parse file info
        line = self.file_list[idx]
        parts = line.split()
        
        # Handle different file formats
        if len(parts) >= 3:
            # Format: sequence frame_id camera
            seq = parts[0]
            stem = int(parts[1])
            side = parts[2]  # 'l' or 'r'
            
            # Map side to camera
            cam = "image_02" if side == "l" else "image_03"
            
            # Get image path
            img_path = os.path.join(self.dataset_path, seq, cam, 'data', f'{stem:010d}.png')
            
            # Check for depth in different possible locations
            depth_path = None
            
            # Try benchmark depth first (most accurate)
            if os.path.exists(os.path.join(self.dataset_path, 'depth_benchmark')):
                bench_depth = os.path.join(self.dataset_path, 'depth_benchmark', seq, 
                                         'proj_depth', 'groundtruth', cam, f'{stem:010d}.png')
                if os.path.exists(bench_depth):
                    depth_path = bench_depth
            
            # Fall back to depth hints if available
            if depth_path is None and os.path.exists(os.path.join(self.dataset_path, 'depth_hints')):
                hints_depth = os.path.join(self.dataset_path, 'depth_hints', seq, cam, f'{stem:010d}.npy')
                if os.path.exists(hints_depth):
                    depth_path = hints_depth
        else:
            # Alternative format: direct paths
            img_path = os.path.join(self.dataset_path, parts[0])
            if len(parts) > 1:
                depth_path = os.path.join(self.dataset_path, parts[1])
            else:
                depth_path = None
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform to image
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Load depth if available
        if depth_path and os.path.exists(depth_path):
            # Handle different depth formats
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
                depth = torch.from_numpy(depth).float().unsqueeze(0)
            else:
                depth_img = Image.open(depth_path)
                if depth_img.mode == 'I':
                    # 16-bit depth maps
                    depth = np.array(depth_img) / 65535.0  # Normalize 16-bit
                else:
                    # 8-bit depth maps
                    depth = np.array(depth_img) / 255.0  # Normalize 8-bit
                
                depth = torch.from_numpy(depth).float().unsqueeze(0)
            
            # Apply transform to depth
            if self.target_transform:
                depth = self.target_transform(depth)
        else:
            # Create empty depth if not available
            depth = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        
        return image, depth


def create_kitti_dataloaders(dataset_path, batch_size=8, transform=None, target_transform=None, split='eigen'):
    """
    Create KITTI dataloaders for training and validation.
    
    Args:
        dataset_path (str): Path to KITTI dataset
        batch_size (int): Batch size
        transform (callable, optional): Transform for input images
        target_transform (callable, optional): Transform for depth maps
        split (str): Split name ('eigen', 'eigen_zhou', or 'benchmark')
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Default transforms if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if target_transform is None:
        target_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        ])
    
    # Create datasets
    train_dataset = KittiDataset(
        dataset_path=dataset_path,
        split=split,
        mode='train',
        transform=transform,
        target_transform=target_transform
    )
    
    val_dataset = KittiDataset(
        dataset_path=dataset_path,
        split=split,
        mode='val',
        transform=transform,
        target_transform=target_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Simple test to check if the dataset works
    import matplotlib.pyplot as plt
    
    # Set up dataset path
    dataset_path = "custom_datasets/kitti/kitti_raw_sync"
    
    # Create a dataset accessor
    accessor = KittiDatasetAccessor(dataset_path, split="eigen", mode="test")
    print(f"Found {len(accessor)} images in the eigen/test split")
    
    # Display first 5 images
    for i, (image, path_id) in enumerate(accessor):
        if i >= 5:
            break
            
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.title(f"Image {i}: {path_id}")
        plt.show()
    
    # Test the PyTorch dataset
    dataset = KittiDataset(dataset_path, split="eigen", mode="train")
    print(f"Found {len(dataset)} samples in the eigen/train split")
    
    # Check first 3 samples
    for i in range(3):
        image, depth = dataset[i]
        print(f"Sample {i}: Image shape: {image.shape}, Depth shape: {depth.shape}")