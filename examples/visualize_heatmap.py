import os
import os.path as osp
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F

from pisl import datasets
from pisl.models import resnet50part
from pisl.utils.data import transforms as T
from pisl.utils.data.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from pisl.utils.serialization import load_checkpoint, copy_state_dict

def get_test_loader(dataset, height, width, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    
    # We can just use query set for some heatmaps
    test_loader = DataLoader(
        Preprocessor(dataset.query, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)
    return test_loader

def visualize_heatmap(model, img_tensor, img_path, save_path):
    # 1. Forward pass to get spatial feature maps (before Global Average Pooling)
    model.eval()
    with torch.no_grad():
        # Get base network output directly [B, C, H, W]
        base_features = model.module.base(img_tensor.cuda())
        
    # 2. Average across channel dimension to get spatial attention map
    # Shape becomes [H, W]
    heatmap = base_features.mean(dim=1).squeeze(0).cpu().numpy()
    
    # Normalize between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    
    # Resize heatmap to match image size using PIL or torch
    # Original image for plotting
    img = Image.open(img_path).convert('RGB')
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    # 3. Create visualization plot
    plt.figure(figsize=(12, 4))
    
    # Plot original
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot raw heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title('Feature Map Attention')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Heatmaps")
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--part', type=int, default=3)
    parser.add_argument('--num-images', type=int, default=5, help="number of images to process")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = datasets.create(args.dataset, args.data_dir)
    # Batch size 1 makes it easier to process individual heatmaps
    test_loader = get_test_loader(dataset, args.height, args.width, 1, args.workers)

    print("Loading model...")
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint, model)

    save_dir = osp.join(working_dir, 'visualizations', 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating {args.num_images} heatmaps...")
    for i, (imgs, fnames, pids, _, _) in enumerate(test_loader):
        if i >= args.num_images:
            break
            
        img_path = fnames[0]
        pid = pids[0].item()
        
        save_path = osp.join(save_dir, f'heatmap_{i}_pid_{pid}.png')
        visualize_heatmap(model, imgs, img_path, save_path)
        print(f"Saved {save_path}")

    print("Done! Check the examples/visualizations/heatmaps/ directory.")

if __name__ == '__main__':
    main()