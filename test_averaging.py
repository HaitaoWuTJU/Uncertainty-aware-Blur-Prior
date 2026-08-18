import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the embeddings safely
file_path = "/work3/s193209/data/ubp_exp/eeg_intra-subject_ubp_EEGProjectLayer_ViT-H-14_noavg/sub-01_seed0/test_embeddings.pt"
data = torch.load(file_path, map_location="cpu", weights_only=False)

eeg_z = data['eeg_z'].float()  # [n_reps, n_images, 1024] or [n_images, n_reps, 1024]
img_z = data['img_z'].float()  # [200, 1024]

# Ensure eeg_z is formatted as: [n_reps, n_images, dim]
if eeg_z.shape[1] == img_z.shape[0] and eeg_z.shape[0] != img_z.shape[0]:
    pass  # Already [n_reps, n_images, dim]
elif eeg_z.shape[0] == img_z.shape[0] and len(eeg_z.shape) == 3:
    eeg_z = eeg_z.permute(1, 0, 2)  # Convert [n_images, n_reps, dim] -> [n_reps, n_images, dim]

total_reps, n_images, embed_dim = eeg_z.shape
max_reps = min(80, total_reps)
n_draws = 10  # 10 random draws per repetition level

# Normalize image targets for cosine similarity
img_z_norm = F.normalize(img_z, p=2, dim=-1)  # [200, 1024]
ground_truth = torch.arange(n_images)         # Correct image index: [0, 1, ..., 199]

top1_means = []
top5_means = []

# 2. Compute retrieval accuracy for k = 1 to max_reps
for k in range(1, max_reps + 1):
    top1_runs = []
    top5_runs = []
    
    for seed in range(n_draws):
        # Set seed for reproducible draws
        np.random.seed(seed + k * 100)
        
        # Sample k unique repetition indices from available repetitions
        selected_idx = np.random.choice(total_reps, size=k, replace=False)
        
        # Average EEG embeddings across the selected repetitions
        eeg_sample_avg = eeg_z[selected_idx].mean(dim=0)  # [200, 1024]
        eeg_sample_norm = F.normalize(eeg_sample_avg, p=2, dim=-1)
        
        # Compute Cosine Similarity Matrix: [200, 200] (EEG queries x Image targets)
        sim_matrix = torch.matmul(eeg_sample_norm, img_z_norm.T)
        
        # Top-1 and Top-5 accuracy
        _, top_indices = sim_matrix.topk(5, dim=-1)  # [200, 5]
        
        # Check if ground_truth is in Top-1 and Top-5
        top1_correct = (top_indices[:, 0] == ground_truth).float().mean().item()
        top5_correct = (top_indices == ground_truth.unsqueeze(1)).any(dim=1).float().mean().item()
        
        top1_runs.append(top1_correct)
        top5_runs.append(top5_correct)
    
    top1_means.append(np.mean(top1_runs) * 100)
    top5_means.append(np.mean(top5_runs) * 100)

# Extract scores for max available repetitions (e.g. 80 reps)
final_t1 = top1_means[-1]
final_t5 = top5_means[-1]

# 3. Plotting
x_reps = np.arange(1, max_reps + 1)

plt.figure(figsize=(9, 5.5), dpi=150)
plt.plot(x_reps, top1_means, label='Top-1 Accuracy', color='#1f77b4', linewidth=2)
plt.plot(x_reps, top5_means, label='Top-5 Accuracy', color='#ff7f0e', linewidth=2)

plt.title(f'Retrieval Accuracy vs EEG Repetitions (10 Draws Avg)\n'
          f'Final ({max_reps} Reps) - Top-1: {final_t1:.2f}% | Top-5: {final_t5:.2f}%', 
          fontsize=12, fontweight='bold')
plt.xlabel('Number of Repetitions Averaged', fontsize=11)
plt.ylabel('Zero-Shot Retrieval Accuracy (%)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(1, max_reps)
plt.ylim(0, 105)
plt.legend(frameon=True, loc='lower right')
plt.tight_layout()

# Save and show
plt.savefig('retrieval_accuracy_vs_reps.png')
plt.show()
print(f"Done! {max_reps} Reps Result -> Top-1: {final_t1:.2f}%, Top-5: {final_t5:.2f}%")
