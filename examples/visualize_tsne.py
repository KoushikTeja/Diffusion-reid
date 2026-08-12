import os
import os.path as osp
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

import torch
from torch import nn

from pisl import datasets
from pisl.models import resnet50part
from pisl.evaluators import extract_all_features
from pisl.utils.data import transforms as T
from pisl.utils.data.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from pisl.utils.serialization import load_checkpoint, copy_state_dict


def get_subset_loader(dataset, height, width, batch_size, workers, num_ids=10):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    # Combine query and gallery
    all_data = list(set(dataset.query) | set(dataset.gallery))

    # Get unique PIDs
    all_pids = list(set([pid for _, pid, _ in all_data]))

    # Select random subset of IDs to plot (plotting all 750 IDs is too messy)
    selected_pids = np.random.choice(all_pids, num_ids, replace=False)

    # Filter dataset for selected IDs
    subset_data = [item for item in all_data if item[1] in selected_pids]

    test_loader = DataLoader(
        Preprocessor(subset_data, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader, selected_pids


def visualize_tsne(features, labels, save_path, perplexity=30, n_iter=1000):
    print("Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    # Generate distinct colors for each ID
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[colors[i]], label=f'ID {label}', alpha=0.8, edgecolors='none', s=50)

    plt.title('t-SNE Visualization of ReID Feature Embeddings', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Person IDs")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize t-SNE")
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--part', type=int, default=3)
    parser.add_argument('--num-ids', type=int, default=15, help="Number of distinct IDs to plot")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset...")
    dataset = datasets.create(args.dataset, args.data_dir)

    # We only take a subset of IDs, otherwise a scatter plot of 750 IDs is unreadable
    test_loader, selected_pids = get_subset_loader(
        dataset, args.height, args.width, args.batch_size, args.workers, num_ids=args.num_ids)

    print("Loading model...")
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint, model)
    model.eval()

    print("Extracting features...")
    # returns dict of {fname: feature_tensor}
    features_g, _, labels_dict = extract_all_features(model, test_loader)

    # Convert dict to array
    fnames = list(features_g.keys())
    features_array = torch.stack([features_g[f] for f in fnames]).numpy()
    labels_array = np.array([labels_dict[f] for f in fnames])

    save_dir = osp.join(working_dir, 'visualizations', args.dataset, 'tsne')
    os.makedirs(save_dir, exist_ok=True)

    save_path = osp.join(save_dir, 'tsne_plot.png')
    visualize_tsne(features_array, labels_array, save_path)
    print(f"Saved {save_path}")
    print("Done! Check the examples/visualizations/tsne/ directory.")


if __name__ == '__main__':
    main()