"""Generates a long path and plots its segments against time."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.numerical.paths import generate_ou_process
from src.numerical.parameters import generate_valid_ou_parameters

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = 2
    delta = 1.0
    N = 8
    K_N = 43
    n_steps_per_block = 20
    block_duration = delta / N
    total_duration = K_N * block_duration
    total_steps = K_N * n_steps_per_block

    theta, mu, sigma = generate_valid_ou_parameters(d, device)

    # Generate one single long path
    long_path = generate_ou_process(
        num_paths=1,
        n_steps=total_steps,
        T=total_duration,
        d=d,
        theta=theta,
        mu=mu,
        sigma=sigma,
        device=device,
        use_time_augmentation=False
    )

    long_path_np = long_path.cpu().numpy().squeeze()
    # Create a common time axis for a single segment
    segment_time_axis = np.linspace(0, block_duration, n_steps_per_block + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('10 Segments of a Single Long Path (Time-Normalized & Origin-Shifted)', fontsize=16)

    # Plot the first 10 segments, each starting from t=0 and (0,0)
    for i in range(10):
        start_idx = i * n_steps_per_block
        end_idx = start_idx + n_steps_per_block + 1
        path_segment = long_path_np[start_idx:end_idx, :]
        # Normalize the segment to start at the origin
        normalized_segment = path_segment - path_segment[0, :]
        ax1.plot(segment_time_axis, normalized_segment[:, 0], label=f'Segment {i+1}')
        ax2.plot(segment_time_axis, normalized_segment[:, 1], label=f'Segment {i+1}')

    ax1.set_ylabel('Dimension 1')
    ax1.grid(True)
    ax1.legend()

    ax2.set_ylabel('Dimension 2')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    ax2.legend()

    plt.savefig('plots/correct_paths_vs_time.png', dpi=300, bbox_inches='tight')
    print('Plot saved to plots/correct_paths_vs_time.png')
