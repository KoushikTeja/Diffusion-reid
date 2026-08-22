import os
import os.path as osp
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn

from pisl import datasets
from pisl.models import resnet50part
from pisl.evaluators import extract_all_features, pairwise_distance
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

    test_set = list(set(dataset.query) | set(dataset.gallery))
    test_loader = DataLoader(
        Preprocessor(test_set, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader


def visualize_ranking(query_info, gallery_infos, save_path, top_k=10):
    q_path, q_pid, q_camid = query_info

    plt.figure(figsize=(16, 4))

    # Plot query
    plt.subplot(1, top_k + 1, 1)
    q_img = Image.open(q_path).convert('RGB')
    plt.imshow(q_img)
    plt.title(f'Query\nID: {q_pid}\nCam: {q_camid}', color='black', fontweight='bold')
    plt.axis('off')

    # Plot gallery results
    for i in range(top_k):
        g_path, g_pid, g_camid = gallery_infos[i]
        plt.subplot(1, top_k + 1, i + 2)
        g_img = Image.open(g_path).convert('RGB')
        plt.imshow(g_img)

        # Color coding: Green for correct (same ID, diff Cam), Red for wrong, Gray for junk (same ID, same Cam)
        if g_pid == q_pid and g_camid == q_camid:
            color = 'gray'
            title = 'Junk'
        elif g_pid == q_pid:
            color = 'green'
            title = 'True'
        else:
            color = 'red'
            title = 'False'

        plt.title(f'{title}\nID: {g_pid}\nCam: {g_camid}', color=color, fontweight='bold')

        # Add colored border
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize ranking")
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--part', type=int, default=3)
    parser.add_argument('--num-queries', type=int, default=5, help="number of query images to visualize")
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Loading dataset...")
    dataset = datasets.create(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    print("Loading model...")
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint, model)
    model.eval()

    print("Extracting features...")
    features_g, _, _ = extract_all_features(model, test_loader)

    print("Computing pairwise distance...")
    dist_m, _, _ = pairwise_distance(features_g, query=dataset.query, gallery=dataset.gallery)
    distmat = dist_m.numpy()

    save_dir = osp.join(working_dir, 'visualizations', args.dataset, 'ranking')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating {args.num_queries} visualizations...")
    # Select random queries
    indices = np.random.choice(len(dataset.query), args.num_queries, replace=False)

    for idx in indices:
        query_info = dataset.query[idx]
        distances = distmat[idx]

        # Sort gallery by distance
        ranked_indices = np.argsort(distances)

        # Get top-15 gallery items (to have enough after potential junks, though we plot top 10)
        top_gallery_infos = [dataset.gallery[i] for i in ranked_indices[:10]]

        save_path = osp.join(save_dir, f'query_{idx}_pid_{query_info[1]}.png')
        visualize_ranking(query_info, top_gallery_infos, save_path, top_k=10)
        print(f"Saved {save_path}")

    print("Done! Check the examples/visualizations/ranking/ directory.")


if __name__ == '__main__':
    main()