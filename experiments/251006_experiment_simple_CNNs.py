import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

# Ensure project root is in sys.path for src.* imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.models import FeatureExtractor
from src.utils.datasets import ChestXrayDataset
from src.utils.utils import set_all_seeds
import joblib
import datetime
import json
import argparse


# ----------------------------------------- Setup ----------------------------------------
# Set device and cudnn tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Set seeds for reproducibility
my_seed = 42
set_all_seeds(my_seed)


# ----------------------------------------- Setup ----------------------------------------
# Parse command-line arguments for configurable parameters
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('--alpha', type=float, default=0.0, help='Fairness penalization factor')
parser.add_argument('--ls', type=int, default=256,
                    help='Size of the intermediate layer before the final classification layer')
parser.add_argument('--bs', type=int, default=32,
                    help='Batch size for training and testing')
args = parser.parse_args()

# Set seeds for reproducibility
my_seed = 42
set_all_seeds(my_seed)

alpha = args.alpha             # Fairness penalization factor
prev2last_layer_size = args.ls # size of the intermediate layer before the final layer
#alpha = 0.0                    # Fairness penalization factor
#prev2last_layer_size = 256     # size of the intermediate layer before the final layer
test_size = 0.2                # proportion of samples for test set
epochs = [10, 100]             # number of epochs per iteration; number of iterations
batch_size = args.bs           # batch size
lr = 1e-4                      # learning rate

main_dir = PROJECT_ROOT
data_subdir = "data/NIH_Chest_XRAY"
img_subdir = "sel_images"
# results
results_subdir = f"results/results_NIH/exp_{prev2last_layer_size}_alpha_{alpha}"
loss_history_log = 'loss_log.json'
# ----------------------------------------------------------------------------------------


# Load data
df = pd.read_csv(os.path.join(main_dir, data_subdir, 'experiment_data.csv'))
image_dir = os.path.join(main_dir, data_subdir, img_subdir)
#print(df.describe())
#print(df.head())

# make sure results directory exists
results_dir = os.path.join(main_dir, results_subdir)
os.makedirs(results_dir, exist_ok=True)

# Train/test split
train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['class'], random_state=my_seed)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# datasets
train_ds = ChestXrayDataset(train_df, image_dir, transform)
test_ds = ChestXrayDataset(test_df, image_dir, transform)

# dataset loader
# Use pin_memory and multiple workers to accelerate host->GPU transfers
loader_kwargs = dict(batch_size=batch_size, pin_memory=True,
                     num_workers=4,
                     persistent_workers=True, prefetch_factor=2)
train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

# create model for feature extraction using a ResNet18 backbone
base_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # load pretrained ResNet18
model = FeatureExtractor(base_resnet, intermediate_size=prev2last_layer_size).to(device)

# Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)




total_epochs = 0
# Load previous loss log if exists
if os.path.exists(os.path.join(results_dir, loss_history_log)):
    with open(os.path.join(results_dir, loss_history_log), 'r') as f:
        loss_log = json.load(f)
        total_epochs = max([int(k) for k in loss_log.keys()])
else:
    loss_log = {}

def fairness_loss(labels, genders, probs, alpha=0.0):
    """
    Calculate fairness penalty (difference in TPR between genders).
    Returns a tensor (in the same device as `probs`) representing the absolute difference in TPR.
    If alpha <= 0.0 or there are not enough positive samples, returns 0.0.
    """
    if alpha <= 0.0:
        return torch.tensor(0.0, device=probs.device)
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        genders_pos = genders[pos_mask]
        probs_pos = probs[pos_mask]
        g1_mask = genders_pos == 0
        g2_mask = genders_pos == 1
        """g1_tpr = (g1_probs > 0.5).float().mean() if g1_probs.numel() > 0 else torch.tensor(0.0, device=probs.device)
        g2_tpr = (g2_probs > 0.5).float().mean() if g2_probs.numel() > 0 else torch.tensor(0.0, device=probs.device)
        return torch.abs(g1_tpr - g2_tpr)"""
        g1_mean = probs_pos[g1_mask].mean() if g1_mask.sum()>0 else torch.tensor(0.0).to(probs.device)
        g2_mean = probs_pos[g2_mask].mean() if g2_mask.sum()>0 else torch.tensor(0.0).to(probs.device)
        return (g1_mean - g2_mean).pow(2)
    else:
        return torch.tensor(0.0, device=probs.device)

def gender_predictability_study(features_all, y, actual_dir_res):
    """Evaluate accuracy and AUC for a model that predicts gender
    from multiple latent representations. Also creates t-SNE plots."""
    gender_pred_results = {}

    for i, features in enumerate(features_all):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(features, y)
        y_pred = clf.predict(features)
        y_prob = clf.predict_proba(features)[:,1]
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred).tolist()
        gender_pred_results[f'layer_{i+1}'] = {'accuracy': acc, 'auc': auc, 'confusion_matrix': cm}
        print(f"Layer {i+1} - Gender prediction AUC: {auc:.3f}")

        """tsne = TSNE(n_components=2, random_state=my_seed)
        X_tsne = tsne.fit_transform(features)
        plt.figure(figsize=(6, 5))
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='coolwarm', alpha=0.6)
        plt.title(f"t-SNE Layer {i+1} (Gender)")
        plt.savefig(os.path.join(actual_dir_res, f"tsne_layer_{i+1}.png"))
        plt.close()"""

    return gender_pred_results

# Fine-tuning function and evaluation
def fine_tune_and_evaluate(epochs=3, alpha=0.0, clip_grad=None):
    """Fine-tune the model and evaluate performance (global and per gender), logging losses and metrics."""
    global total_epochs
    epoch_losses = {'train': [], 'val': []}

    # Training loop
    for epoch in range(epochs):
        model.train()
        num_correct_predictions = 0.0
        num_train_samples = 0.0
        running_loss = 0.0
        running_ce_loss = 0.0
        running_fair_penalty = 0.0
        running_loss = 0.0
        running_grad_norms = 0.0
        running_grad_ce_norms = 0.0

        # per batch loop
        for images, labels, genders in train_loader:
            # use non_blocking transfer from pinned memory
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            genders = genders.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:,1]

            preds = outputs.argmax(dim=1)
            num_correct_predictions += (preds == labels).sum().item()
            num_train_samples += labels.size(0)

            loss_ce = criterion(outputs, labels)
            fairness_penalty = fairness_loss(labels, genders, probs, alpha)
            running_ce_loss += loss_ce.item()
            running_fair_penalty += fairness_penalty.item()
            loss = loss_ce + alpha * fairness_penalty  # Add the fairness penalty to the loss
            running_loss += loss.item()

            #loss.backward()

            # --- diagnostic gradients ---
            # 1) Gradients from CE only
            optimizer.zero_grad()
            loss_ce.backward(retain_graph=True)
            grad_norm_ce = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_ce += p.grad.norm().item() ** 2
            running_grad_ce_norms += grad_norm_ce ** 0.5

            # 2) Gradients from total loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm_total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_total += p.grad.norm().item() ** 2
            running_grad_norms += grad_norm_total ** 0.5

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

        train_loss = running_loss / len(train_loader)
        train_acc = num_correct_predictions / num_train_samples
        grad_norms = running_grad_norms / len(train_loader)
        grad_ce_norms = running_grad_ce_norms / len(train_loader)
        epoch_losses['train'].append({
            'loss': train_loss.item() if hasattr(train_loss, 'item') else train_loss,
            'accuracy': train_acc.item() if hasattr(train_acc, 'item') else train_acc,
            "grad_norm": grad_norms.item() if hasattr(grad_norms, 'item') else grad_norms,
            "grad_ce_norm": grad_ce_norms.item() if hasattr(grad_ce_norms, 'item') else grad_ce_norms
        })#.append({'loss': train_loss, 'accuracy': train_acc}))
        print(f"Loss:  {running_ce_loss/ len(train_loader):.4f} + {alpha:.1f}*"+
              f"fairness penalty: {running_fair_penalty/ len(train_loader):.4f}"+
              f" # Loss gradient norms: {epoch_losses['train'][-1]['grad_norm']:.4f}"+
              f" # CE-Loss grad. norms: {epoch_losses['train'][-1]['grad_ce_norm']:.4f}")

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        correct_val = 0.0
        total_val = 0.0
        with torch.no_grad():
            for images, labels, genders in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                genders = genders.to(device, non_blocking=True)

                outputs = model(images)
                probs = F.softmax(outputs, dim=1)[:,1]

                # loss calculation
                loss_ce = criterion(outputs, labels)
                fairness_penalty = fairness_loss(labels, genders, probs, alpha)
                val_loss = loss_ce + alpha * fairness_penalty  # validation loss with fairness penalty

                # TP and total samples for accuracy calculation
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_loss = val_loss / len(test_loader)
        val_acc = correct_val / total_val
        #epoch_losses['val'].append({'loss': val_loss, 'accuracy': val_acc})
        epoch_losses['val'].append({
            'loss': val_loss.item() if hasattr(val_loss, 'item') else val_loss,
            'accuracy': val_acc.item() if hasattr(val_acc, 'item') else val_acc
        })

        print(f"Epoch {total_epochs + epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Update total epochs and log losses
    total_epochs += epochs
    loss_log[str(total_epochs)] = epoch_losses
    with open(os.path.join(results_dir, loss_history_log), 'w') as f:
        json.dump(loss_log, f, indent=2)

    # Create directory for current-epoch-number results
    actual_dir_res = os.path.join(results_dir, f'e_{total_epochs}')
    os.makedirs(actual_dir_res, exist_ok=True)

    # Evaluation by gender
    model.eval()
    all_preds, all_labels, all_genders = [], [], []
    num_layers = len(model.features_all)
    #all_features = [[] for _ in range(num_layers)]
    all_features_tensors = [[] for _ in range(num_layers)]
    with torch.no_grad():
        for images, labels, genders in test_loader:
            images = images.to(device, non_blocking=True)
            # keep labels/genders on CPU copy for quick numpy ops
            labels_cpu = labels.numpy() if not labels.is_cuda else labels.cpu().numpy()
            genders_cpu = genders.numpy() if not genders.is_cuda else genders.cpu().numpy()

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_cpu)
            all_genders.extend(genders_cpu)

            # collect feature tensors (keep on device)
            for i in range(num_layers):
                #all_features[i].append(model.features_all[i].cpu().numpy())
                # detach to avoid storing autograd graph
                all_features_tensors[i].append(model.features_all[i].detach())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_genders = np.array(all_genders)
    #all_features = [np.vstack(f) for f in all_features]
    all_features = [torch.cat(l, dim=0).cpu().numpy() if len(l) > 0 else np.empty((0, 0)) for l in all_features_tensors]

    result_summary = {}
    for group, name in [(None, 'All'), (0, 'Male'), (1, 'Female')]:
        if group is None:
            mask = np.ones_like(all_genders, dtype=bool)
        else:
            mask = (all_genders == group)
        acc = accuracy_score(all_labels[mask], all_preds[mask])
        auc = roc_auc_score(all_labels[mask], all_preds[mask])
        cm = confusion_matrix(all_labels[mask], all_preds[mask]).tolist()
        result_summary[name] = {'accuracy': acc, 'auc': auc, 'confusion_matrix': cm}
        print(f"{name} - Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    gender_pred_results = gender_predictability_study(all_features, all_genders, actual_dir_res)

    joblib.dump(result_summary, os.path.join(actual_dir_res, f"eval_results.pkl"))
    joblib.dump(gender_pred_results, os.path.join(actual_dir_res, f"latent_gender_results.pkl"))

# Run fine-tuning and evaluation for specified number of iterations, each with specified epochs
# In each iteration, results are stored in a separate subdirectory
for _ in range(epochs[1]):
    fine_tune_and_evaluate(epochs=epochs[0], alpha=alpha)

