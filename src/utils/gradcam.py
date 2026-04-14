import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM for a FeatureExtractor model targeting its last conv layer (layer4)."""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = [
            model.layer4.register_forward_hook(self._save_activation),
            model.layer4.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, images, target_class=1):
        """Compute Grad-CAM heatmaps for a batch of images.

        Parameters
        ----------
        images : Tensor (N, C, H, W)
        target_class : int
            Class index to explain (default 1 = disease positive).

        Returns
        -------
        heatmaps : np.ndarray (N, H_img, W_img) in [0, 1]
        """
        self.model.eval()
        images.requires_grad_(True)
        outputs = self.model(images)

        self.model.zero_grad()
        target = outputs[:, target_class]
        target.sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)             # (N, H, W)
        cam = F.relu(cam)

        # Normalize per sample and resize to image dimensions
        h, w = images.shape[2], images.shape[3]
        heatmaps = []
        for i in range(cam.shape[0]):
            hm = cam[i]
            hm = hm - hm.min()
            if hm.max() > 0:
                hm = hm / hm.max()
            hm = F.interpolate(hm.unsqueeze(0).unsqueeze(0), size=(h, w),
                               mode='bilinear', align_corners=False)
            heatmaps.append(hm.squeeze().cpu().numpy())
        return np.stack(heatmaps)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def collect_gradcam_by_gender(model, dataloader, device, target_class=1):
    """Run Grad-CAM on a dataloader and group heatmaps by gender.

    Returns
    -------
    dict with keys 'male' and 'female', each containing a list of dicts:
        {'heatmap': np.ndarray, 'image': np.ndarray, 'label': int, 'pred': int}
    """
    gc = GradCAM(model)
    results = {'male': [], 'female': []}

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.enable_grad():
        for images, labels, genders in dataloader:
            images = images.to(device)
            heatmaps = gc(images, target_class=target_class)
            preds = model(images).argmax(dim=1).cpu().numpy()

            images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            images_np = (images_np * std + mean).clip(0, 1)

            for i in range(len(labels)):
                entry = {
                    'heatmap': heatmaps[i],
                    'image': images_np[i],
                    'label': labels[i].item(),
                    'pred': preds[i],
                }
                if genders[i].item() == 0:
                    results['male'].append(entry)
                else:
                    results['female'].append(entry)

    gc.remove_hooks()
    return results


def plot_average_heatmaps(gradcam_results, save_path=None):
    """Plot average Grad-CAM heatmaps for male vs female and their difference."""
    male_maps = np.stack([r['heatmap'] for r in gradcam_results['male']])
    female_maps = np.stack([r['heatmap'] for r in gradcam_results['female']])

    avg_male = male_maps.mean(axis=0)
    avg_female = female_maps.mean(axis=0)
    diff = avg_female - avg_male

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = [
        f'Hombres (n={len(male_maps)})',
        f'Mujeres (n={len(female_maps)})',
        'Diferencia (Mujeres - Hombres)',
    ]
    maps = [avg_male, avg_female, diff]
    cmaps = ['jet', 'jet', 'RdBu_r']

    for ax, hm, title, cmap in zip(axes, maps, titles, cmaps):
        vmin, vmax = (None, None) if cmap != 'RdBu_r' else (-abs(diff).max(), abs(diff).max())
        im = ax.imshow(hm, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Grad-CAM promedio por género', fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sample_heatmaps(gradcam_results, n_samples=4, save_path=None):
    """Plot individual Grad-CAM examples side by side for each gender."""
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

    for row, (gender, label) in enumerate([('male', 'Hombre'), ('female', 'Mujer')]):
        samples = gradcam_results[gender][:n_samples]
        for col, s in enumerate(samples):
            ax = axes[row, col]
            ax.imshow(s['image'])
            ax.imshow(s['heatmap'], cmap='jet', alpha=0.4)
            status = 'TP' if s['label'] == 1 and s['pred'] == 1 else \
                     'FP' if s['label'] == 0 and s['pred'] == 1 else \
                     'FN' if s['label'] == 1 and s['pred'] == 0 else 'TN'
            ax.set_title(f'{label} — {status}')
            ax.axis('off')

    fig.suptitle('Ejemplos Grad-CAM por género', fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
