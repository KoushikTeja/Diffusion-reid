from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from pisl import datasets
from pisl.loss import DiffusionThetaLoss
import maximum_mean_discrepancy
from pisl.models import resnet50part
from pisl.loss import CameraContrast
from pisl.trainers import PISLTrainerCAM
from pisl.evaluators import Evaluator, extract_all_features
from pisl.utils.data import IterLoader
from pisl.utils.data import transforms as T
from pisl.utils.data.sampler import RandomMultipleGallerySampler
from pisl.utils.data.preprocessor import Preprocessor
from pisl.utils.logging import Logger
from pisl.utils.faiss_rerank import compute_ranked_list, compute_jaccard_distance
from pisl.utils.serialization import load_checkpoint, copy_state_dict

best_mAP = 0


def get_data(name, data_dir):
    return datasets.create(name, data_dir)


def get_train_loader(dataset, height, width, batch_size, workers, num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    sampler = RandomMultipleGallerySampler(train_set, num_instances) if rmgs_flag else None
    train_loader = IterLoader(
        DataLoader(
            Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=not rmgs_flag,
            pin_memory=True,
            drop_last=True
        ),
        length=iters
    )
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader


def compute_pseudo_labels(features, cluster, k1):
    mat_dist = compute_jaccard_distance(features, k1=k1, k2=6)
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids


def compute_semantic_consistency(features_g, features_p, k, search_option=0):
    print("Compute semantic consistency score...")
    N, D, P = features_p.size()
    score = torch.zeros(N, P, device=features_g.device)
    end = time.time()

    ranked_list_g = compute_ranked_list(features_g, k=k, search_option=search_option, verbose=False)
    gb_neighbors_all = torch.stack([features_g[ranked_list_g[j]] for j in range(N)])

    initial_batch_size = 32
    min_batch_size = 16

    for i in range(P):
        ranked_list_p_i = compute_ranked_list(features_p[:, :, i], k=k, search_option=search_option, verbose=False)

        for batch_start in range(0, N, initial_batch_size):
            batch_end = min(batch_start + initial_batch_size, N)
            batch_size_actual = batch_end - batch_start

            pt_neighbors_batch = torch.stack([
                features_p[:, :, i][ranked_list_p_i[j]] for j in range(batch_start, batch_end)
            ])
            gb_neighbors_batch = gb_neighbors_all[batch_start:batch_end]

            for j in range(batch_size_actual):
                gb_dist = torch.cdist(gb_neighbors_batch[j], gb_neighbors_batch[j])
                pt_dist = torch.cdist(pt_neighbors_batch[j], pt_neighbors_batch[j])

                gb_dist = gb_dist / (gb_dist.max() + 1e-8)
                pt_dist = pt_dist / (pt_dist.max() + 1e-8)

                gb_dist_flat = gb_dist.view(-1)
                pt_dist_flat = pt_dist.view(-1)

                mmd_score = maximum_mean_discrepancy.mmd_loss(
                    gb_dist_flat.unsqueeze(0),
                    pt_dist_flat.unsqueeze(0)
                ) / (k * 2)

                mmd_score = torch.clamp(mmd_score, min=0.0, max=1.0)
                score[batch_start + j, i] = mmd_score

    print("semantic consistency score time cost: {}".format(time.time() - end))
    return score


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)


def main_worker(args):
    global best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                     testset=sorted(dataset.train))

    num_parts = args.part
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)

    # ── Resume from checkpoint ──────────────────────────────────
    if args.resume:
        print(f"=> Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint successfully")

    if args.best_mAP > 0:
        best_mAP = args.best_mAP
        print(f"=> Restored best mAP: {best_mAP:.1%}")
    # ────────────────────────────────────────────────────────────

    evaluator = Evaluator(model)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # ── Fast-forward lr_scheduler to start_epoch ────────────────
    for _ in range(args.start_epoch):
        lr_scheduler.step()
    # ────────────────────────────────────────────────────────────

    consistency_score_log = torch.FloatTensor([])

    # ── Start from start_epoch instead of 0 ─────────────────────
    for epoch in range(args.start_epoch, args.epochs):
    # ────────────────────────────────────────────────────────────
        global_features, part_features, _ = extract_all_features(model, cluster_loader)
        global_features = torch.cat([global_features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        part_features = torch.cat([part_features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

        if epoch == args.start_epoch:
            cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=8)

        pseudo_labels, num_classes = compute_pseudo_labels(global_features, cluster, args.k1)
        consistency_scores = compute_semantic_consistency(global_features, part_features, k=args.knn)
        consistency_score_log = torch.cat([consistency_score_log, consistency_scores.unsqueeze(0)], dim=0)

        num_outliers = 0
        new_dataset = []
        sample_indices, camera_ids, person_ids = [], [], []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            pid = label.item()
            if pid >= num_classes:
                num_outliers += 1
            else:
                new_dataset.append((fname, pid, cid))
                sample_indices.append(i)
                camera_ids.append(cid)
                person_ids.append(pid)

        train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size,
                                        args.workers, args.num_instances, args.iters, trainset=new_dataset)

        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(epoch, num_classes, num_outliers))

        sample_indices = np.asarray(sample_indices)
        camera_ids = np.asarray(camera_ids)
        person_ids = np.asarray(person_ids)
        global_features = global_features[sample_indices, :]
        part_features = part_features[sample_indices, :, :]
        consistency_scores = consistency_scores[sample_indices, :]

        global_centroids, part_centroids = [], []
        camera_proxies, camera_part_proxies, proxy_pids, proxy_cids = [], [], [], []
        for pid in sorted(np.unique(person_ids)):
            pid_indices = np.where(person_ids == pid)[0]
            global_centroids.append(global_features[pid_indices].mean(0))
            part_centroids.append(part_features[pid_indices].mean(0))

            for cid in sorted(np.unique(camera_ids[pid_indices])):
                cid_indices = np.where(camera_ids == cid)[0]
                common_indices = np.intersect1d(pid_indices, cid_indices)
                camera_proxies.append(global_features[common_indices].mean(0))
                camera_part_proxies.append(part_features[common_indices].mean(0))
                proxy_pids.append(pid)
                proxy_cids.append(cid)

        global_centroids = F.normalize(torch.stack(global_centroids), p=2, dim=1)
        model.module.classifier.weight.data[:num_classes].copy_(global_centroids)
        camera_memory = CameraContrast(global_centroids.size(1), len(proxy_pids)).cuda()
        camera_memory.proxy = F.normalize(torch.stack(camera_proxies), p=2, dim=1).cuda()
        camera_memory.pids = torch.Tensor(proxy_pids).long().cuda()
        camera_memory.cids = torch.Tensor(proxy_cids).long().cuda()

        part_memories = []
        for i in range(num_parts):
            part_centroids_i = torch.stack(part_centroids)[:, :, i]
            part_centroids_i = F.normalize(part_centroids_i, p=2, dim=1)
            part_classifier = getattr(model.module, 'classifier' + str(i))
            part_classifier.weight.data[:num_classes].copy_(part_centroids_i)

            part_memory = CameraContrast(global_centroids.size(1), len(proxy_pids)).cuda()
            camera_part_proxies_i = torch.stack(camera_part_proxies)[:, :, i]
            part_memory.proxy = F.normalize(camera_part_proxies_i, p=2, dim=1).cuda()
            part_memory.pids = torch.Tensor(proxy_pids).long().cuda()
            part_memory.cids = torch.Tensor(proxy_cids).long().cuda()
            part_memories.append(part_memory)

        trainer = PISLTrainerCAM(model, consistency_scores, camera_memory, part_memories,
                               num_class=num_classes, num_part=num_parts,
                               Wref=args.Wref, se=args.se, Wcam=args.Wcam, Wdiff=args.Wdiff)

        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))
        lr_scheduler.step()

        if ((epoch+1) % args.eval_step == 0) or (epoch == args.epochs-1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
            print('\n* Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}\n'.format(epoch, mAP, best_mAP))

    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))
    np.save(osp.join(args.logs_dir, 'scores.npy'), consistency_score_log.numpy())

    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unsupervised Learning of Intrinsic Semantics With Diffusion Model for Person Re-Identification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs/test'))

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)

    # resume
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help="path to checkpoint to resume from")
    parser.add_argument('--start-epoch', type=int, default=0,
                        help="epoch to start from when resuming")
    parser.add_argument('--best-mAP', type=float, default=0.0,
                        help="best mAP so far, used when resuming")

    # PISL
    parser.add_argument('--part', type=int, default=3)
    parser.add_argument('--knn', type=int, default=20)
    parser.add_argument('--Wref', type=float, default=0.5)
    parser.add_argument('--se', type=int, default=5)
    parser.add_argument('--Wcam', type=float, default=0.5)
    parser.add_argument('--Wdiff', type=float, default=0.1)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # cluster
    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--eps', type=float, default=0.5)

    main()