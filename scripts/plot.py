import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

rows = list(csv.DictReader(open('/root/demo3/exp/results/results.csv')))
data = {r['name']: r for r in rows}
OUT = '/root/demo3/exp'

# ---------- 图 1: LoRA rank vs F1 ----------
ranks = [8, 16, 32]
f1s = [float(data['rank8']['f1']), float(data['baseline']['f1']), float(data['rank32']['f1'])]
plt.figure(figsize=(6,4))
plt.plot(ranks, f1s, 'o-', color='steelblue', linewidth=2, markersize=8)
for x, y in zip(ranks, f1s):
    plt.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0,8), ha='center')
plt.xlabel('LoRA rank'); plt.ylabel('F1')
plt.title('F1 vs LoRA rank (QLoRA, lr=5e-5)')
plt.xticks(ranks); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rank_f1.png', dpi=150)
plt.close()

# ---------- 图 2: Learning rate vs F1 ----------
lrs = ['1e-5', '5e-5', '1e-4']
f1s = [float(data['lr1e5']['f1']), float(data['baseline']['f1']), float(data['lr1e4']['f1'])]
plt.figure(figsize=(6,4))
plt.plot(lrs, f1s, 's-', color='coral', linewidth=2, markersize=8)
for x, y in zip(lrs, f1s):
    plt.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0,8), ha='center')
plt.xlabel('Learning rate'); plt.ylabel('F1')
plt.title('F1 vs Learning rate (QLoRA, rank=16)')
plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_lr_f1.png', dpi=150)
plt.close()

# ---------- 图 3: QLoRA vs LoRA (F1 + 显存) ----------
fig, ax1 = plt.subplots(figsize=(6,4))
configs = ['QLoRA (4bit)', 'LoRA (fp16)']
f1_vals = [float(data['baseline']['f1']), float(data['lora']['f1'])]
mem_vals = [16.6, 23.4]
x = [0, 1]
bars = ax1.bar(x, f1_vals, width=0.4, color=['steelblue', 'coral'], label='F1')
ax1.set_ylabel('F1'); ax1.set_ylim(0.8, 0.87)
ax1.set_xticks(x); ax1.set_xticklabels(configs)
for i, v in enumerate(f1_vals):
    ax1.text(i, v + 0.002, f'{v:.4f}', ha='center')

ax2 = ax1.twinx()
ax2.plot(x, mem_vals, 'D--', color='green', markersize=10, label='GPU Mem (GB)')
ax2.set_ylabel('GPU Memory (GB)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
for i, v in enumerate(mem_vals):
    ax2.text(i, v + 0.5, f'{v:.1f}G', ha='center', color='green')

plt.title('QLoRA vs LoRA: F1 & GPU Memory')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_quant_compare.png', dpi=150)
plt.close()

print("Saved: fig_rank_f1.png, fig_lr_f1.png, fig_quant_compare.png")
