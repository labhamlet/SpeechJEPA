import torch 
import time 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from einops import rearrange

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'pdf.fonttype': 42,
})

def _get_feat_extract_output_lengths(input_lengths, cfg):
    """
    Computes the output length of the convolutional layers
    """
    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

    for kernel_size, stride in zip(cfg["conv_kernel"], cfg["conv_stride"]):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
    return input_lengths


def measure_batches_per_second(
    loader,
    warmup_batches: int = 5,
    benchmark_batches: int = 50,
    log_every: int = 10,
) -> float:
    """
    Measures the throughput of a dataloader in batches per second.

    Args:
        loader:            Any iterable dataloader where batch[0] is the input tensor.
        warmup_batches:    Number of batches to skip before timing starts.
        benchmark_batches: Number of batches to time.
        log_every:         Print a progress line every N batches.

    Returns:
        Samples per second (float).
    """
    total_samples = 0
    batch_count = 0
    start_time = None

    for i, batch in enumerate(loader):
        # --- Warmup ---
        if i < warmup_batches:
            print(f"[Warmup {i+1}/{warmup_batches}] shape: {batch[0].shape}")
            continue

        # --- Start timer after warmup ---
        if start_time is None:
            start_time = time.perf_counter()

        batch_count += 1

        if batch_count % log_every == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"[Benchmark {batch_count}/{benchmark_batches}] "
                f"shape: {batch[0].shape} | "
                f"batches/sec: {batch_count / elapsed:.1f} | "
                f"elapsed: {elapsed:.1f}s"
            )

        if batch_count >= benchmark_batches:
            break

    elapsed = time.perf_counter() - start_time
    sps = total_samples / elapsed

    print("\n=== Benchmark Results ===")
    print(f"  Batches      : {batch_count}")
    print(f"  Total samples: {total_samples}")
    print(f"  Elapsed      : {elapsed:.2f}s")
    print(f"  Batches/sec  : {sps:.1f}")

    return sps


def visualize_masks(
    n_times: int,
    in_channels: int,
    ctx_mask, 
    tgt_mask,
    n_targets_per_context: int,
    save_path: str = None,
    figsize: tuple = (3.5, 2.2)
):
    if torch.is_tensor(ctx_mask):
        ctx_mask = ctx_mask.cpu().numpy()
        tgt_mask = tgt_mask.cpu().numpy()

    ctx_mask = rearrange(ctx_mask, "B (S C) -> B C S", C=in_channels)
    tgt_mask = rearrange(tgt_mask, "B N (S C) -> B N C S", C=in_channels)

    batch_idx = 0
    #Here, we flip the mask, this effectively flips the masked indices to False, and non masked ones to True
    #In the real mask "True" means to apply the attention mask.
    #Every True element is set to the green color
    ctx_indices_sample = ~ctx_mask[batch_idx]
    #We do not flip the targets, because targets are actually masked in the JEPA training, 
    #Everything else is also masked except the context block
    tgt_mask_sample = tgt_mask[batch_idx]

    CONTEXT_CODE = 0
    TARGET_CODE = 1
    MASKED_CODE = 2

    full_matrix = np.full((n_targets_per_context, n_times), MASKED_CODE)
    c = 0
    for i in range(n_targets_per_context):
        full_matrix[i, ctx_indices_sample[c]] = CONTEXT_CODE
        full_matrix[i, tgt_mask_sample[i, c]] = TARGET_CODE

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#a9d48f', '#0089af', '#e0e0e0']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    BAR_HEIGHT = 1.0
    GAP_SIZE = 0.35

    y_ticks_pos = []
    y_tick_labels = []

    # X coordinates for pcolormesh (edges of the cells)
    x_grid = np.arange(0, n_times + 1)

    for i in range(n_targets_per_context):
        row_data = full_matrix[i].reshape(1, -1)

        y_start = i * (BAR_HEIGHT + GAP_SIZE)
        y_end = y_start + BAR_HEIGHT

        y_ticks_pos.append(y_start + (BAR_HEIGHT / 2))
        y_tick_labels.append(f"T{i+1}")

        y_grid = np.array([y_start, y_end])

        # Use pcolormesh instead of imshow
        # edgecolors='face' prevents tiny white lines between blocks in PDF
        # linewidth=0 ensures no borders
        ax.pcolormesh(
            x_grid,
            y_grid,
            row_data,
            cmap=cmap,
            norm=norm,
            shading='flat',
            edgecolors='face',
            linewidth=0,
            snap=True
        )

    ax.set_xlim(0, n_times)
    ax.set_ylim(n_targets_per_context * (BAR_HEIGHT + GAP_SIZE), -GAP_SIZE)

    ax.set_yticks(y_ticks_pos)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel("Prediction Target")
    ax.set_xlabel("Time Step ($t$)")
    ax.set_xticks(np.linspace(0, n_times, 5).astype(int))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(axis='y', length=0)

    legend_patches = [
        Patch(facecolor=colors[0], label='Context', edgecolor='0.3', linewidth=0.5),
        Patch(facecolor=colors[1], label='Target', edgecolor='0.3', linewidth=0.5),
        Patch(facecolor=colors[2], label='Masked', edgecolor='0.3', linewidth=0.5),
    ]

    ax.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0
    )

    plt.subplots_adjust(top=0.8, bottom=0.20, left=0.15, right=0.98)

    if save_path:
        plt.savefig(save_path + ".pdf", format='pdf', dpi=600, bbox_inches='tight')

