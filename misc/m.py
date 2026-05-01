import json
import matplotlib.pyplot as plt

with open('proposed_solution/RL_agent/training_log.json', 'r') as f:
    log = json.load(f)

# Extract training data
train_epochs = [e['epoch'] for e in log['train_losses']]
losses = [e['loss'] for e in log['train_losses']]
rewards = [e['avg_reward'] for e in log['train_losses']]

# Extract val data
val_epochs = [e['epoch'] for e in log['val_results']]
val_overall = [e['overall_hit1'] * 100 for e in log['val_results']]
val_single = [e['single_hop_hit1'] * 100 for e in log['val_results']]
val_multi = [e['multi_hop_hit1'] * 100 for e in log['val_results']]

best_epoch = log['best_epoch']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 14))

# ── Plot 1: Training Loss ─────────────────────────────
ax1.plot(train_epochs, losses, color='#1a3a5c', linewidth=2, marker='o', markersize=4)
ax1.axvline(x=best_epoch, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Best model (epoch {best_epoch})')
ax1.set_ylabel('Loss', fontsize=13)
ax1.set_title('Training Loss over Epochs', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Plot 2: Average Reward ────────────────────────────
ax2.plot(train_epochs, rewards, color='#2ecc71', linewidth=2, marker='o', markersize=4)
ax2.axvline(x=best_epoch, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Best model (epoch {best_epoch})')
ax2.set_ylabel('Average Reward', fontsize=13)
ax2.set_title('Average Reward over Epochs', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Plot 3: Validation Hit@1 ──────────────────────────
ax3.plot(val_epochs, val_overall, color='#1a3a5c', linewidth=2, marker='o', markersize=4, label='Overall Hit@1')
ax3.plot(val_epochs, val_single, color='#E8874A', linewidth=2, marker='s', markersize=4, label='Single-hop Hit@1')
ax3.plot(val_epochs, val_multi, color='#2ecc71', linewidth=2, marker='^', markersize=4, label='Multi-hop Hit@1')
ax3.axvline(x=best_epoch, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Best model (epoch {best_epoch})')
ax3.set_xlabel('Epoch', fontsize=13)
ax3.set_ylabel('Hit@1 (%)', fontsize=13)
ax3.set_title('Validation Hit@1 over Epochs', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
ax3.set_axisbelow(True)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout(pad=3.0)
plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
plt.show()