import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from datetime import datetime
import uuid

# Import our models
from models.mdec_unet import create_mdec_unet_model
from kitti_dataset_helper import KittiDataset

def create_dataloaders(dataset_path, batch_size=8, val_split=0.1, transform=None, target_transform=None, split='eigen'):
    """Create dataloaders for KITTI dataset."""
    # Set up transforms
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
    
    # Create KITTI datasets
    train_dataset = KittiDataset(dataset_path, split=split, mode='train', 
                                transform=transform, target_transform=target_transform)
    val_dataset = KittiDataset(dataset_path, split=split, mode='val', 
                              transform=transform, target_transform=target_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def depth_loss(pred, target, valid_mask=None):
    """Scale-invariant loss function for depth estimation."""
    if valid_mask is None:
        valid_mask = (target > 0).detach()
    
    # Apply mask to predictions and targets
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # L1 loss
    l1_loss = torch.abs(pred_valid - target_valid).mean()
    
    # Scale-invariant MSE term
    diff = torch.log(pred_valid) - torch.log(target_valid)
    si_mse = torch.mean(diff**2) - 0.5 * (torch.mean(diff))**2
    
    # Edge-aware smoothness loss
    edges = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    smoothness = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]) * torch.exp(-edges)
    smoothness_loss = smoothness.mean()
    
    # Combine losses
    loss = 0.85 * l1_loss + 0.1 * si_mse + 0.05 * smoothness_loss
    
    return loss

def validate(model, val_loader, device):
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    abs_rel_error = 0.0
    sq_rel_error = 0.0
    rmse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images, targets = images.to(device), targets.to(device)
            valid_mask = (targets > 0).detach()
            
            # Forward pass
            outputs = model(images)
            
            # Get main disparity output (scale 0)
            disparity = outputs['disp'][0]
            
            # Calculate loss
            loss = depth_loss(disparity, targets, valid_mask)
            val_loss += loss.item()
            
            # Calculate metrics
            pred_valid = disparity[valid_mask]
            target_valid = targets[valid_mask]
            
            if pred_valid.numel() > 0:
                # Absolute relative error
                abs_rel = torch.abs(pred_valid - target_valid) / target_valid
                abs_rel_error += abs_rel.mean().item()
                
                # Squared relative error
                sq_rel = ((pred_valid - target_valid) ** 2) / target_valid
                sq_rel_error += sq_rel.mean().item()
                
                # RMSE
                rmse += torch.sqrt(torch.mean((pred_valid - target_valid) ** 2)).item()
                
                num_samples += 1
    
    # Average metrics
    val_loss /= len(val_loader)
    if num_samples > 0:
        abs_rel_error /= num_samples
        sq_rel_error /= num_samples
        rmse /= num_samples
    
    return val_loss, abs_rel_error, sq_rel_error, rmse

def train_model(args):
    """Train the MDEC-compatible UNet model."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_mdec_unet_model(
        n_channels=3,
        out_scales=(0, 1, 2, 3),
        bilinear=True,
        base_channels=args.base_channels
    )
    model.to(device)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        args.dataset_path,
        batch_size=args.batch_size,
        split=args.split
    )
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create model directory
    model_id = f"mdec_unet_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    model_dir = os.path.join(args.output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Training
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, targets = images.to(device), targets.to(device)
            valid_mask = (targets > 0).detach()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss (main scale)
            loss = depth_loss(outputs['disp'][0], targets, valid_mask)
            
            # Add multi-scale loss if desired
            if args.multi_scale_loss:
                for scale in [1, 2, 3]:
                    if scale in outputs['disp']:
                        scale_factor = 2 ** scale
                        scaled_targets = nn.functional.interpolate(
                            targets, 
                            scale_factor=1/scale_factor, 
                            mode='nearest'
                        )
                        scaled_mask = nn.functional.interpolate(
                            valid_mask.float(), 
                            scale_factor=1/scale_factor, 
                            mode='nearest'
                        ).bool()
                        
                        scale_loss = depth_loss(outputs['disp'][scale], scaled_targets, scaled_mask)
                        loss += 0.5 ** scale * scale_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation
        val_loss, abs_rel, sq_rel, val_rmse = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Metrics - AbsRel: {abs_rel:.4f}, SqRel: {sq_rel:.4f}, RMSE: {val_rmse:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
            
            # Save model info
            model_info = {
                'name': f"MDEC UNet (KITTI {args.split})",
                'type': 'mdec_unet',
                'input_size': [256, 256],
                'channels': args.base_channels,
                'metrics': {
                    'abs_rel': abs_rel,
                    'sq_rel': sq_rel,
                    'rmse': val_rmse,
                    'val_loss': val_loss
                },
                'training': {
                    'dataset': args.dataset_path,
                    'split': args.split,
                    'epochs': epoch + 1,
                    'batch_size': args.batch_size,
                    'lr': args.lr
                },
                'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': f"MDEC-compatible UNet model trained on KITTI {args.split} dataset"
            }
            
            with open(os.path.join(model_dir, 'info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"Saved best model (val_loss: {val_loss:.4f}) to {model_dir}")
    
    print(f"Training completed. Best model saved to {model_dir}")
    return model_dir

def main():
    parser = argparse.ArgumentParser(description="Train MDEC-compatible UNet model on KITTI dataset")
    parser.add_argument("--dataset_path", default="custom_datasets/kitti/kitti_raw_sync", 
                       help="Path to KITTI dataset")
    parser.add_argument("--split", default="eigen", choices=["eigen", "eigen_zhou", "benchmark"], 
                       help="KITTI dataset split to use")
    parser.add_argument("--output_dir", default="custom_models/trained", 
                       help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--base_channels", type=int, default=64, 
                       help="Number of base channels in UNet")
    parser.add_argument("--multi_scale_loss", action="store_true", 
                       help="Use multi-scale loss during training")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA training")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    model_dir = train_model(args)
    print(f"Model saved to: {model_dir}")

if __name__ == "__main__":
    main()