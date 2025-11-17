import torch
from torch.utils.data import Dataset, DistributedSampler
import h5py
import numpy as np
from pathlib import Path
import json


class ShardedHDF5Dataset(Dataset):
    """
    Dataset for sharded HDF5 files optimized for multi-node training.
    Each worker preferentially reads from local shards when available.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        split: str = "train",          # "train" or "test"
        train_fraction: float = 0.9,   # fraction of episodes in train
        split_seed: int = 42,          # seed for reproducible split
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.train_fraction = train_fraction
        self.split_seed = split_seed

        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.num_shards = self.metadata['num_shards']
        self.shard_files = [
            self.data_dir / f"shard_{i:04d}.h5"
            for i in range(self.num_shards)
        ]

        # Build window index across all shards
        self.windows = []
        self.episode_lengths = []  # Store all episode lengths for analysis

        for shard_idx, shard_file in enumerate(self.shard_files):
            with h5py.File(shard_file, 'r') as f:
                num_episodes = f.attrs['num_episodes']
                lengths = f['episode_lengths'][:]

                # Store episode lengths for statistics
                self.episode_lengths.extend(lengths.tolist())

                for ep_idx, ep_length in enumerate(lengths):
                    for start in range(0, ep_length - window_size + 1, stride):
                        self.windows.append((shard_idx, ep_idx, start))

        # Collect all (shard_idx, ep_idx) pairs
        all_episodes = sorted({(shard_idx, ep_idx) for shard_idx, ep_idx, _ in self.windows})

        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(all_episodes))

        num_train_eps = int(self.train_fraction * len(all_episodes))
        train_eps = {all_episodes[i] for i in perm[:num_train_eps]}
        test_eps  = {all_episodes[i] for i in perm[num_train_eps:]}

        self.split_info = {
            "train_episodes": sorted(list(train_eps)),
            "test_episodes": sorted(list(test_eps)),
        }

        if self.split == "train":
            keep = train_eps
        elif self.split == "test":
            keep = test_eps
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Filter windows based on chosen split
        self.windows = [w for w in self.windows if (w[0], w[1]) in keep]
        print(f"{self.split.capitalize()} split: {len(self.windows)} windows "
              f"from {len(keep)} episodes")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        shard_idx, ep_idx, start = self.windows[idx]
        end = start + self.window_size

        shard_file = self.shard_files[shard_idx]

        # Open HDF5 file (each worker maintains its own handle)
        with h5py.File(shard_file, 'r') as f:
            images = f['images'][ep_idx, start:end]
            actions = f['actions'][ep_idx, start:end]

        # Convert to PyTorch
        images = torch.from_numpy(images).float() / 255.0
        images = images.permute(0, 3, 1, 2)
        actions = torch.from_numpy(actions)
        
        return {'image': images, 'action': actions}


    def get_episode_length_statistics(self):
        """
        Calculate comprehensive statistics about episode lengths.
        
        Returns:
            dict with statistics about episode lengths
        """
        lengths = np.array(self.episode_lengths)
        
        stats = {
            'total_episodes': len(lengths),
            'total_timesteps': int(np.sum(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'mean_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'std_length': float(np.std(lengths)),
            'percentile_25': float(np.percentile(lengths, 25)),
            'percentile_75': float(np.percentile(lengths, 75)),
            'percentile_90': float(np.percentile(lengths, 90)),
            'percentile_95': float(np.percentile(lengths, 95)),
            'percentile_99': float(np.percentile(lengths, 99)),
        }
        
        return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze ShardedHDF5Dataset and print statistics"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/scratch/ja5009/soar_data_sharded/',
        help='Path to sharded HDF5 directory'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=96,
        help='Window size for sliding windows'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Stride for sliding windows'
    )
    parser.add_argument(
        '--show_histogram',
        action='store_true',
        help='Show histogram of episode lengths (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ShardedHDF5Dataset Analysis")
    print("="*70)
    print(f"\nLoading dataset from: {args.data_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}\n")
    
    # Create dataset
    dataset = ShardedHDF5Dataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
    )
    
    # Get statistics
    stats = dataset.get_episode_length_statistics()
    
    print("\n" + "="*70)
    print("Episode Length Statistics")
    print("="*70)
    print(f"Total Episodes:           {stats['total_episodes']:,}")
    print(f"Total Timesteps:          {stats['total_timesteps']:,}")
    print(f"Total Windows:            {len(dataset):,}")
    print(f"\nLength Statistics:")
    print(f"  Min:                    {stats['min_length']:.0f} steps")
    print(f"  Max:                    {stats['max_length']:.0f} steps")
    print(f"  Mean:                   {stats['mean_length']:.2f} steps")
    print(f"  Median:                 {stats['median_length']:.2f} steps")
    print(f"  Std Dev:                {stats['std_length']:.2f} steps")
    print(f"\nPercentiles:")
    print(f"  25th percentile:        {stats['percentile_25']:.0f} steps")
    print(f"  75th percentile:        {stats['percentile_75']:.0f} steps")
    print(f"  90th percentile:        {stats['percentile_90']:.0f} steps")
    print(f"  95th percentile:        {stats['percentile_95']:.0f} steps")
    print(f"  99th percentile:        {stats['percentile_99']:.0f} steps")
    
    # Calculate storage efficiency
    avg_windows_per_episode = len(dataset) / stats['total_episodes']
    print(f"\nDataset Efficiency:")
    print(f"  Avg windows per episode: {avg_windows_per_episode:.2f}")
    print(f"  Window coverage:         {avg_windows_per_episode * args.stride / stats['mean_length'] * 100:.1f}%")
    
    # Shard information
    print(f"\nShard Information:")
    print(f"  Number of shards:        {dataset.num_shards}")
    print(f"  Avg episodes per shard:  {stats['total_episodes'] / dataset.num_shards:.1f}")
    
    # Calculate approximate memory usage per batch
    if 'image_shape' in dataset.metadata:
        img_shape = dataset.metadata['image_shape']
        bytes_per_window = (
            args.window_size * img_shape[0] * img_shape[1] * img_shape[2] * 4  # float32
        )
        print(f"\nMemory Usage (per window):")
        print(f"  Image shape:             {img_shape}")
        print(f"  Bytes per window:        {bytes_per_window / (1024**2):.2f} MB")
        print(f"  Batch of 5 windows:      {5 * bytes_per_window / (1024**2):.2f} MB")
    
    print("\n" + "="*70)
    
    # Optional: Show histogram
    if args.show_histogram:
        try:
            import matplotlib.pyplot as plt
            
            lengths = np.array(dataset.episode_lengths)
            
            plt.figure(figsize=(12, 6))
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(stats['mean_length'], color='r', linestyle='--', 
                       label=f"Mean: {stats['mean_length']:.1f}")
            plt.axvline(stats['median_length'], color='g', linestyle='--', 
                       label=f"Median: {stats['median_length']:.1f}")
            plt.xlabel('Episode Length (timesteps)')
            plt.ylabel('Frequency')
            plt.title('Episode Length Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot(lengths, vert=True)
            plt.ylabel('Episode Length (timesteps)')
            plt.title('Episode Length Box Plot')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_file = Path(args.data_dir) / 'episode_length_analysis.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nHistogram saved to: {output_file}")
            
            plt.show()
            
        except ImportError:
            print("\nWarning: matplotlib not installed. Cannot show histogram.")
            print("Install with: pip install matplotlib")
    
    # Test loading a sample
    print("\nTesting data loading...")
    try:
        sample = dataset[0]
        print(f"  Sample shapes:")
        print(f"    Images: {sample['image'].shape}")
        print(f"    Actions: {sample['action'].shape}")
        print(f"  Sample dtypes:")
        print(f"    Images: {sample['image'].dtype}")
        print(f"    Actions: {sample['action'].dtype}")
        print(f"  Image value range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print("  ✓ Data loading successful!")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
    
    print("\n" + "="*70)
