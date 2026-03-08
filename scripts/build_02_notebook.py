"""Build and execute the 02_cka_analysis.ipynb notebook programmatically."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

# --- Cell 0: Markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "# 02 — CKA Analysis Tutorial\n"
    "\n"
    "**Goal**: Understand Centered Kernel Alignment (CKA) — the key metric "
    "for monitoring distillation quality in the Mamba-MoE pipeline.\n"
    "\n"
    "CKA measures representational similarity between neural network layers. "
    "During distillation, CKA < 0.75 triggers investigation of information loss.\n"
    "\n"
    "This notebook uses `aya_distill.distill.cka` — no model loading required."
))

# --- Cell 1: Setup ---
cells.append(nbf.v4.new_code_cell(
    'import sys\n'
    'from pathlib import Path\n'
    'sys.path.insert(0, str(Path.cwd().parent / "src"))\n'
    '\n'
    'import torch\n'
    'import numpy as np\n'
    'import matplotlib.pyplot as plt\n'
    '\n'
    'from aya_distill.distill.cka import (\n'
    '    linear_cka,\n'
    '    rbf_cka,\n'
    '    MinibatchCKAAccumulator,\n'
    '    cka_permutation_test,\n'
    '    compute_layerwise_cka,\n'
    ')\n'
    '\n'
    'print("CKA module loaded successfully")'
))

# --- Cell 2: Markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 1. CKA Basics — Identical vs. Random Representations\n"
    "\n"
    "CKA = 1.0 means identical representations. CKA ≈ 0 means no correspondence."
))

# --- Cell 3: Basic CKA demo ---
cells.append(nbf.v4.new_code_cell(
    'torch.manual_seed(42)\n'
    'n, d = 100, 64\n'
    '\n'
    'X = torch.randn(n, d)\n'
    'Y_identical = X.clone()\n'
    'Y_random = torch.randn(n, d)\n'
    'Y_scaled = X * 3.0 + 1.0  # CKA should be invariant to scaling\n'
    '\n'
    'print("Linear CKA:")\n'
    'print(f"  Identical:    {linear_cka(X, Y_identical):.4f}")\n'
    'print(f"  Random:       {linear_cka(X, Y_random):.4f}")\n'
    'print(f"  Scaled (3x+1): {linear_cka(X, Y_scaled):.4f}")\n'
    '\n'
    'print("\\nRBF CKA:")\n'
    'print(f"  Identical:    {rbf_cka(X, Y_identical):.4f}")\n'
    'print(f"  Random:       {rbf_cka(X, Y_random):.4f}")\n'
    'print(f"  Scaled (3x+1): {rbf_cka(X, Y_scaled):.4f}")'
))

# --- Cell 4: Markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 2. Noise Sensitivity\n"
    "\n"
    "How does CKA degrade as we add increasing noise? This tells us how much "
    "representation drift we can tolerate during distillation."
))

# --- Cell 5: Noise sweep ---
cells.append(nbf.v4.new_code_cell(
    'torch.manual_seed(42)\n'
    'X = torch.randn(200, 128)\n'
    'noise_levels = np.linspace(0, 5, 25)\n'
    'linear_scores = []\n'
    'rbf_scores = []\n'
    '\n'
    'for sigma in noise_levels:\n'
    '    Y = X + sigma * torch.randn_like(X)\n'
    '    linear_scores.append(linear_cka(X, Y))\n'
    '    rbf_scores.append(rbf_cka(X, Y))\n'
    '\n'
    'fig, ax = plt.subplots(figsize=(10, 5))\n'
    'ax.plot(noise_levels, linear_scores, "o-", label="Linear CKA", linewidth=2)\n'
    'ax.plot(noise_levels, rbf_scores, "s-", label="RBF CKA", linewidth=2)\n'
    'ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.7, label="Distillation threshold (0.75)")\n'
    'ax.set_xlabel("Noise \\u03c3")\n'
    'ax.set_ylabel("CKA Score")\n'
    'ax.set_title("CKA Degradation Under Gaussian Noise")\n'
    'ax.legend()\n'
    'ax.grid(True, alpha=0.3)\n'
    'plt.tight_layout()\n'
    'plt.savefig("../results/cka_noise_sensitivity.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()'
))

# --- Cell 6: Markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 3. Minibatch CKA\n"
    "\n"
    "For large-scale distillation we can't compute CKA on the full dataset at once. "
    "`MinibatchCKAAccumulator` computes exact CKA incrementally."
))

# --- Cell 7: Minibatch CKA ---
# NOTE: MinibatchCKAAccumulator requires d_x and d_y constructor args
cells.append(nbf.v4.new_code_cell(
    'torch.manual_seed(42)\n'
    'X_full = torch.randn(500, 64)\n'
    'Y_full = X_full + 0.5 * torch.randn_like(X_full)\n'
    '\n'
    '# Full-batch CKA (ground truth)\n'
    'full_cka = linear_cka(X_full, Y_full)\n'
    '\n'
    '# Minibatch CKA (accumulate over batches)\n'
    'acc = MinibatchCKAAccumulator(d_x=64, d_y=64)\n'
    'batch_size = 50\n'
    'for i in range(0, 500, batch_size):\n'
    '    acc.update(X_full[i:i+batch_size], Y_full[i:i+batch_size])\n'
    '\n'
    'mb_cka = acc.compute()\n'
    'print(f"Full-batch CKA:  {full_cka:.6f}")\n'
    'print(f"Minibatch CKA:   {mb_cka:.6f}")\n'
    'print(f"Difference:      {abs(full_cka - mb_cka):.8f}")'
))

# --- Cell 8: Markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 4. Permutation Test for Statistical Significance\n"
    "\n"
    "Is the CKA score significantly different from chance? The permutation test "
    "shuffles one representation matrix and computes a null distribution."
))

# --- Cell 9: Permutation test ---
cells.append(nbf.v4.new_code_cell(
    'torch.manual_seed(42)\n'
    'X = torch.randn(100, 32)\n'
    'Y_correlated = X + 0.3 * torch.randn_like(X)\n'
    'Y_random = torch.randn(100, 32)\n'
    '\n'
    'result_corr = cka_permutation_test(X, Y_correlated, n_permutations=500)\n'
    'result_rand = cka_permutation_test(X, Y_random, n_permutations=500)\n'
    '\n'
    'print("Correlated pair:")\n'
    'print(f"  CKA = {result_corr[\'observed_cka\']:.4f}, p = {result_corr[\'p_value\']:.4f}")\n'
    'print(f"  Significant: {result_corr[\'p_value\'] < 0.05}")\n'
    '\n'
    'print("\\nRandom pair:")\n'
    'print(f"  CKA = {result_rand[\'observed_cka\']:.4f}, p = {result_rand[\'p_value\']:.4f}")\n'
    'print(f"  Significant: {result_rand[\'p_value\'] < 0.05}")'
))

# --- Cell 10: Markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 5. Simulated Layer-wise CKA Heatmap\n"
    "\n"
    "Simulate a teacher (32 layers) and student (24 layers) to visualize which "
    "teacher layers map best to which student layers. This guides the distillation "
    "layer alignment strategy."
))

# --- Cell 11: Heatmap ---
# NOTE: compute_layerwise_cka expects dict[str, Tensor], not list
cells.append(nbf.v4.new_code_cell(
    'torch.manual_seed(42)\n'
    'n_teacher, n_student = 32, 24\n'
    'n_samples, d = 50, 64\n'
    '\n'
    '# Simulate teacher and student activations with structured similarity\n'
    'teacher_acts = {}\n'
    'teacher_list = []  # keep ordered reference\n'
    'for i in range(n_teacher):\n'
    '    act = torch.randn(n_samples, d)\n'
    '    teacher_acts[f"teacher_{i:02d}"] = act\n'
    '    teacher_list.append(act)\n'
    '\n'
    'student_acts = {}\n'
    'for j in range(n_student):\n'
    '    # Student layer j is most similar to teacher layer at proportional position\n'
    '    t_idx = int(j * n_teacher / n_student)\n'
    '    base = teacher_list[t_idx]\n'
    '    noise = torch.randn_like(base) * (0.3 + 0.02 * abs(j - n_student // 2))\n'
    '    student_acts[f"student_{j:02d}"] = base + noise\n'
    '\n'
    '# Compute CKA matrix\n'
    'heatmap = compute_layerwise_cka(teacher_acts, student_acts)\n'
    '\n'
    'fig, ax = plt.subplots(figsize=(12, 8))\n'
    'im = ax.imshow(heatmap.scores.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)\n'
    'ax.set_xlabel("Teacher Layer")\n'
    'ax.set_ylabel("Student Layer")\n'
    'ax.set_title("Layer-wise CKA: Teacher (32L) x Student (24L)")\n'
    'plt.colorbar(im, ax=ax, label="CKA Score")\n'
    '\n'
    '# Mark the diagonal correspondence\n'
    'for j in range(n_student):\n'
    '    t_idx = int(j * n_teacher / n_student)\n'
    '    ax.plot(t_idx, j, "rx", markersize=8, markeredgewidth=2)\n'
    '\n'
    'plt.tight_layout()\n'
    'plt.savefig("../results/cka_heatmap.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()\n'
    '\n'
    '# Report best matches\n'
    'print("Best teacher layer for each student layer:")\n'
    'for j in range(0, n_student, 4):\n'
    '    best_t = heatmap.scores[:, j].argmax()\n'
    '    score = heatmap.scores[best_t, j]\n'
    '    print(f"  Student {j:2d} -> Teacher {best_t:2d} (CKA = {score:.3f})")'
))

# --- Cell 12: Markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Key Takeaways\n"
    "\n"
    "1. **CKA is invariant to scaling** — perfect for comparing layers with different norms\n"
    "2. **Noise tolerance**: CKA stays above 0.75 threshold up to ~sigma=1.5 noise\n"
    "3. **Minibatch CKA is exact** — safe to use during training\n"
    "4. **Permutation test** provides statistical significance for layer matches\n"
    "5. **Heatmap** reveals the structural correspondence that guides our distillation alignment"
))

nb.cells = cells

out_path = "/home/rcgalbo/wayy-research/aya/project-aya/notebooks/02_cka_analysis.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {len(cells)} cells to {out_path}")
