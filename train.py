"""Train MedViTV2-tiny on DermaMNIST with different loss functions (ELEC-444 final project).

Same model, data, augmentation and optimizer as the MedViTV2 repo. Only the loss changes.
Differences from the repo's main.py (on purpose):
  * the best epoch is chosen on the VALIDATION set (balanced accuracy), never the test set
  * the test set is evaluated once, at the end, with the best checkpoint
  * training resumes automatically after a Colab disconnect

Examples:
  python train.py --loss ce --seed 0
  python train.py --loss weighted --weighting sqrt --seed 0
  python train.py --loss focal --gamma 2 --seed 0
"""
import argparse, csv, json, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)

MELANOMA = 4  # class index of melanoma in DermaMNIST


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--loss', choices=['ce', 'weighted', 'focal'], required=True)
    p.add_argument('--weighting', choices=['inverse', 'sqrt'], default='inverse', help='for --loss weighted')
    p.add_argument('--gamma', type=float, default=2.0, help='for --loss focal')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--out_root', default='/content/drive/MyDrive/elec444_project/results')
    p.add_argument('--workers', type=int, default=2)
    return p.parse_args()


def run_name(a):
    if a.loss == 'ce':
        tag = 'ce'
    elif a.loss == 'weighted':
        tag = f'weighted_{a.weighting}'
    else:
        tag = f'focal_g{a.gamma:g}'
    return f'{tag}_seed{a.seed}'


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_datasets(root):
    os.makedirs(root, exist_ok=True)
    info = INFO['dermamnist']
    DataClass = getattr(medmnist, info['python_class'])
    # Same transforms as the repo's datasets.build_transform
    train_tf = T.Compose([T.RandomResizedCrop(224), T.AugMix(alpha=0.4), T.RandomHorizontalFlip(p=0.4),
                          T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])
    eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])
    ds = {s: DataClass(split=s, transform=train_tf if s == 'train' else eval_tf, download=True,
                       as_rgb=True, root=root, size=224, mmap_mode='r') for s in ['train', 'val', 'test']}
    classes = [info['label'][str(i)] for i in range(len(info['label']))]
    return ds, classes


def build_model(num_classes):
    from MedViT import MedViT_tiny
    return MedViT_tiny(num_classes=num_classes)


def build_loss(a, train_labels, num_classes, device):
    if a.loss == 'ce':
        return nn.CrossEntropyLoss()
    if a.loss == 'weighted':
        from losses import make_weighted_ce
        return make_weighted_ce(train_labels, num_classes=num_classes, scheme=a.weighting, device=device)
    try:
        from losses import make_focal_loss
    except ImportError:
        sys.exit('ERROR: losses.py has no make_focal_loss yet (Person C adds it).')
    return make_focal_loss(gamma=a.gamma, device=device)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(model(x.to(device, non_blocking=True)).float().softmax(-1).cpu())
        ys.append(y.view(-1))
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def metrics(probs, y, classes):
    pred = probs.argmax(1)
    k = len(classes)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
    try:
        auc = roc_auc_score(y, probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
    return {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'macro_auc': auc,
        'macro_f1': float(f.mean()),
        'melanoma_recall': float(r[MELANOMA]),
        'per_class': {c: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(s[i])}
                      for i, c in enumerate(classes)},
        'confusion_matrix': confusion_matrix(y, pred, labels=list(range(k))).tolist(),
    }


def main():
    a = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = run_name(a)
    out = os.path.join(a.out_root, name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'test_metrics.json')):
        sys.exit(f'{name} is already finished (test_metrics.json exists). Delete its folder to rerun it.')
    print(f'Run: {name}\nSaving to: {out}', flush=True)

    set_seed(a.seed)
    ds, classes = build_datasets(a.data_root)
    train_labels = ds['train'].labels.ravel()
    g = torch.Generator(); g.manual_seed(a.seed)
    train_loader = DataLoader(ds['train'], batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                              pin_memory=True, generator=g)
    val_loader = DataLoader(ds['val'], batch_size=128, shuffle=False, num_workers=a.workers, pin_memory=True)
    test_loader = DataLoader(ds['test'], batch_size=128, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = build_model(len(classes)).to(device)
    loss_fn = build_loss(a, train_labels, len(classes), device)
    # Same optimizer and schedule as the repo's main.py (scheduler stepped every batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.05)
    t_max = a.epochs * len(ds['train']) // a.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=5e-6)

    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({**vars(a), 'run_name': name, 'model': 'MedViT_tiny',
                   'selection': 'best val balanced_accuracy'}, f, indent=2)

    # ----- resume after a disconnect -----
    last_path, best_path, log_path = (os.path.join(out, n) for n in ['last.pth', 'best.pth', 'log.csv'])
    start_epoch, best_score = 0, -1.0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start_epoch, best_score = ck['epoch'] + 1, ck['best_score']
        torch.set_rng_state(ck['rng']).cpu(); g.manual_seed(a.seed + start_epoch)
        print(f'Resuming from epoch {start_epoch + 1}', flush=True)

    fields = ['epoch', 'train_loss', 'val_accuracy', 'val_balanced_accuracy', 'val_macro_auc',
              'val_melanoma_recall', 'lr', 'minutes']
    if start_epoch == 0:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(fields)

    for epoch in range(start_epoch, a.epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.view(-1).long().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step(); sched.step()
            total += loss.item() * len(y); n += len(y)

        vm = metrics(*predict(model, val_loader, device), classes)
        mins = (time.time() - t0) / 60
        row = [epoch + 1, total / n, vm['accuracy'], vm['balanced_accuracy'], vm['macro_auc'],
               vm['melanoma_recall'], sched.get_last_lr()[0], mins]
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        improved = vm['balanced_accuracy'] > best_score
        if improved:
            best_score = vm['balanced_accuracy']
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_balanced_accuracy': best_score}, best_path)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': epoch, 'best_score': best_score, 'rng': torch.get_rng_state()}, last_path)
        print(f'[{name}] epoch {epoch + 1}/{a.epochs}  loss {total / n:.3f}  val acc {vm["accuracy"]:.3f}  '
              f'val bal-acc {vm["balanced_accuracy"]:.3f}  val melanoma recall {vm["melanoma_recall"]:.3f}  '
              f'({mins:.1f} min){"  * best" if improved else ""}', flush=True)

    # ----- final evaluation: best-on-validation checkpoint, test set used once -----
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best['model'])
    val_probs, val_y = predict(model, val_loader, device)
    test_probs, test_y = predict(model, test_loader, device)
    np.save(os.path.join(out, 'test_probs.npy'), test_probs)
    np.save(os.path.join(out, 'test_labels.npy'), test_y)
    np.save(os.path.join(out, 'val_probs.npy'), val_probs)
    result = {'run_name': name, 'best_epoch': best['epoch'] + 1,
              'val': metrics(val_probs, val_y, classes), 'test': metrics(test_probs, test_y, classes)}
    with open(os.path.join(out, 'test_metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    if os.path.exists(last_path):
        os.remove(last_path)  # keep Drive small; best.pth stays

    t = result['test']
    print(f'\n===== FINISHED {name} (best epoch {result["best_epoch"]}) =====')
    print(f'TEST  accuracy {t["accuracy"]:.4f} | balanced accuracy {t["balanced_accuracy"]:.4f} | '
          f'macro AUC {t["macro_auc"]:.4f} | macro F1 {t["macro_f1"]:.4f} | melanoma recall {t["melanoma_recall"]:.4f}')
    for c, m in t['per_class'].items():
        print(f'  {c[:40]:40s}  P {m["precision"]:.3f}  R {m["recall"]:.3f}  F1 {m["f1"]:.3f}  (n={m["support"]})')
    print(f'\nFiles saved in {out}')


if __name__ == '__main__':
    main()
