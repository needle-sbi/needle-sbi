#!/usr/bin/env python3
"""
Plotting Utilities for PseudoModel Benchmark Analysis

Creates:
- 3D scatter plots showing all three dimensions
- 2D projection plots (3 combinations)
- 1D slice plots (3 dimensions)
- Comparison plots for different strategies
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Optional seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("seaborn not found, using default matplotlib styling")
    plt.style.use('ggplot')

# Configure plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


class BenchmarkAnalyzer:
    """Analyze and visualize PseudoModel benchmark results."""
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing benchmark result JSON files
        """
        self.results_dir = Path(results_dir)
        self.data = None
        self.strategies = ['sequential', 'parallel', 'vectorized']
        
    def load_results(self) -> pd.DataFrame:
        """
        Load all benchmark results into a DataFrame.
        
        Returns:
            DataFrame with columns: strategy, n_networks, batch_size, 
                                   n_network_params, time_ms, ...
        """
        all_results = []
        
        # Find all JSON files
        json_files = sorted(self.results_dir.glob("benchmark_job_*.json"))
        
        print(f"Found {len(json_files)} result files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                param_set = data['metadata']['param_set']
                
                for result in data['results']:
                    row = {
                        'strategy': result['strategy'],
                        'n_networks': param_set['n_networks'],
                        'num_ensembles': param_set['num_ensembles'],
                        'folds': param_set['folds'],
                        'batch_size': param_set['batch_size'],
                        'n_network_params': param_set['n_network_params'],
                        'network_size_name': param_set['network_size_name'],
                        'hidden_dim': param_set['hidden_dim'],
                        'n_hidden': param_set['n_hidden'],
                        'time_ms': result['total_time'],
                    }
                    
                    # Add breakdown components
                    for component, time_val in result['breakdown'].items():
                        row[f'breakdown_{component}'] = time_val
                    
                    all_results.append(row)
                    
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        self.data = pd.DataFrame(all_results)
        
        print(f"Loaded {len(self.data)} benchmark results")
        print(f"Strategies: {self.data['strategy'].unique()}")
        print(f"Parameter ranges:")
        print(f"  n_networks: {self.data['n_networks'].min()} - {self.data['n_networks'].max()}")
        print(f"  batch_size: {self.data['batch_size'].min()} - {self.data['batch_size'].max()}")
        print(f"  n_network_params: {self.data['n_network_params'].min()} - {self.data['n_network_params'].max()}")
        
        return self.data
    
    def plot_3d_scatter(self, strategy: str = 'sequential', output_path: str = None):
        """
        Create 3D scatter plot for a specific strategy.
        
        Args:
            strategy: Which strategy to plot
            output_path: Path to save figure (optional)
        """
        if self.data is None:
            self.load_results()
        
        # Filter data
        data_subset = self.data[self.data['strategy'] == strategy].copy()
        
        if len(data_subset) == 0:
            print(f"No data for strategy: {strategy}")
            return
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot with color mapping by time
        scatter = ax.scatter(
            data_subset['n_networks'],
            data_subset['batch_size'],
            data_subset['n_network_params'],
            c=data_subset['time_ms'],
            cmap='viridis',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Labels
        ax.set_xlabel('Number of Networks', fontsize=12, labelpad=10)
        ax.set_ylabel('Batch Size', fontsize=12, labelpad=10)
        ax.set_zlabel('Network Parameters', fontsize=12, labelpad=10)
        ax.set_title(f'PseudoModel Performance - {strategy.upper()} Strategy', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Execution Time (ms)', fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to: {output_path}")
        
        return fig
    
    def plot_2d_projections(self, strategy: str = 'sequential', output_path: str = None):
        """
        Create 2D projection plots for all dimension pairs.
        
        Args:
            strategy: Which strategy to plot
            output_path: Path to save figure (optional)
        """
        if self.data is None:
            self.load_results()
        
        data_subset = self.data[self.data['strategy'] == strategy].copy()
        
        if len(data_subset) == 0:
            print(f"No data for strategy: {strategy}")
            return
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Define dimension pairs
        pairs = [
            ('n_networks', 'batch_size', 'Number of Networks', 'Batch Size'),
            ('n_networks', 'n_network_params', 'Number of Networks', 'Network Parameters'),
            ('batch_size', 'n_network_params', 'Batch Size', 'Network Parameters'),
        ]
        
        for ax, (x_col, y_col, x_label, y_label) in zip(axes, pairs):
            scatter = ax.scatter(
                data_subset[x_col],
                data_subset[y_col],
                c=data_subset['time_ms'],
                cmap='viridis',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time (ms)', fontsize=10)
        
        fig.suptitle(f'2D Projections - {strategy.upper()} Strategy', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved 2D projections to: {output_path}")
        
        return fig
    
    def plot_1d_slices(self, strategy: str = 'sequential', output_path: str = None):
        """
        Create 1D slice plots showing behavior along each dimension.
        
        Args:
            strategy: Which strategy to plot
            output_path: Path to save figure (optional)
        """
        if self.data is None:
            self.load_results()
        
        data_subset = self.data[self.data['strategy'] == strategy].copy()
        
        if len(data_subset) == 0:
            print(f"No data for strategy: {strategy}")
            return
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Define dimensions
        dimensions = [
            ('n_networks', 'Number of Networks'),
            ('batch_size', 'Batch Size'),
            ('n_network_params', 'Network Parameters'),
        ]
        
        for ax, (dim_col, dim_label) in zip(axes, dimensions):
            # Group by dimension and calculate statistics
            grouped = data_subset.groupby(dim_col)['time_ms'].agg(['mean', 'std', 'min', 'max'])
            
            x_vals = grouped.index.values
            y_mean = grouped['mean'].values
            y_std = grouped['std'].values
            
            # Plot mean with error bars
            ax.errorbar(x_vals, y_mean, yerr=y_std, fmt='o-', 
                       capsize=5, capthick=2, linewidth=2, markersize=8,
                       label='Mean ± Std')
            
            # Fill between min/max
            ax.fill_between(x_vals, grouped['min'].values, grouped['max'].values,
                           alpha=0.2, label='Min-Max Range')
            
            ax.set_xlabel(dim_label, fontsize=11)
            ax.set_ylabel('Execution Time (ms)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        fig.suptitle(f'1D Behavior Along Each Dimension - {strategy.upper()} Strategy',
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved 1D slices to: {output_path}")
        
        return fig
    
    def plot_strategy_comparison(self, output_path: str = None):
        """
        Create comparison plots across all strategies.
        
        Args:
            output_path: Directory to save figures (optional)
        """
        if self.data is None:
            self.load_results()
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 1. Overall distribution
        ax = axes[0]
        for strategy in self.strategies:
            subset = self.data[self.data['strategy'] == strategy]
            if len(subset) > 0:
                ax.hist(subset['time_ms'], bins=30, alpha=0.5, label=strategy.upper())
        ax.set_xlabel('Execution Time (ms)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Execution Time Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. By number of networks
        ax = axes[1]
        for strategy in self.strategies:
            subset = self.data[self.data['strategy'] == strategy]
            if len(subset) > 0:
                grouped = subset.groupby('n_networks')['time_ms'].mean()
                ax.plot(grouped.index, grouped.values, 'o-', label=strategy.upper(), 
                       linewidth=2, markersize=8)
        ax.set_xlabel('Number of Networks', fontsize=11)
        ax.set_ylabel('Mean Execution Time (ms)', fontsize=11)
        ax.set_title('Performance vs Network Count', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. By batch size
        ax = axes[2]
        for strategy in self.strategies:
            subset = self.data[self.data['strategy'] == strategy]
            if len(subset) > 0:
                grouped = subset.groupby('batch_size')['time_ms'].mean()
                ax.plot(grouped.index, grouped.values, 'o-', label=strategy.upper(),
                       linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_ylabel('Mean Execution Time (ms)', fontsize=11)
        ax.set_title('Performance vs Batch Size', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. By network size
        ax = axes[3]
        for strategy in self.strategies:
            subset = self.data[self.data['strategy'] == strategy]
            if len(subset) > 0:
                grouped = subset.groupby('network_size_name')['time_ms'].mean()
                # Sort by parameter count
                size_order = ['small', 'medium', 'large', 'xlarge']
                x_positions = range(len(grouped))
                ax.bar([p + 0.25 * self.strategies.index(strategy) for p in x_positions],
                      grouped.values, width=0.25, label=strategy.upper(), alpha=0.7)
        ax.set_xlabel('Network Size', fontsize=11)
        ax.set_ylabel('Mean Execution Time (ms)', fontsize=11)
        ax.set_title('Performance vs Network Size', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Strategy Comparison Across All Dimensions',
                     fontsize=16, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        
        if output_path:
            save_path = Path(output_path) / "strategy_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")
        
        return fig
    
    def generate_summary_statistics(self, output_path: str = None):
        """
        Generate summary statistics table.
        
        Args:
            output_path: Path to save CSV file (optional)
        """
        if self.data is None:
            self.load_results()
        
        summary = self.data.groupby('strategy')['time_ms'].agg([
            'count', 'mean', 'std', 'min', 'max', 
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).round(2)
        
        print("\nSummary Statistics (all in ms):")
        print("=" * 80)
        print(summary)
        print("=" * 80)
        
        if output_path:
            summary.to_csv(output_path)
            print(f"\nSaved summary statistics to: {output_path}")
        
        return summary
    
    def create_all_plots(self, output_dir: str):
        """
        Generate all plots and save to output directory.
        
        Args:
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("Generating All Plots")
        print("=" * 80 + "\n")
        
        # Load data
        self.load_results()
        
        # Summary statistics
        print("\n1. Generating summary statistics...")
        self.generate_summary_statistics(
            output_path / "summary_statistics.csv"
        )
        
        # For each strategy
        for strategy in self.strategies:
            print(f"\n2. Generating plots for {strategy.upper()} strategy...")
            
            # 3D plot
            self.plot_3d_scatter(
                strategy=strategy,
                output_path=output_path / f"{strategy}_3d.png"
            )
            plt.close()
            
            # 2D projections
            self.plot_2d_projections(
                strategy=strategy,
                output_path=output_path / f"{strategy}_2d_projections.png"
            )
            plt.close()
            
            # 1D slices
            self.plot_1d_slices(
                strategy=strategy,
                output_path=output_path / f"{strategy}_1d_slices.png"
            )
            plt.close()
        
        # Strategy comparison
        print("\n3. Generating strategy comparison plots...")
        self.plot_strategy_comparison(output_path=output_path)
        plt.close()
        
        print("\n" + "=" * 80)
        print(f"All plots saved to: {output_path}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize PseudoModel benchmark results"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing benchmark JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save plots'
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['all', '3d', '2d', '1d', 'comparison', 'stats'],
        default='all',
        help='Type of plots to generate'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['sequential', 'parallel', 'vectorized'],
        default='sequential',
        help='Strategy to plot (for single strategy plots)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BenchmarkAnalyzer(args.results_dir)
    
    # Generate requested plots
    if args.plot_type == 'all':
        analyzer.create_all_plots(args.output_dir)
    elif args.plot_type == '3d':
        analyzer.plot_3d_scatter(
            strategy=args.strategy,
            output_path=Path(args.output_dir) / f"{args.strategy}_3d.png"
        )
    elif args.plot_type == '2d':
        analyzer.plot_2d_projections(
            strategy=args.strategy,
            output_path=Path(args.output_dir) / f"{args.strategy}_2d.png"
        )
    elif args.plot_type == '1d':
        analyzer.plot_1d_slices(
            strategy=args.strategy,
            output_path=Path(args.output_dir) / f"{args.strategy}_1d.png"
        )
    elif args.plot_type == 'comparison':
        analyzer.plot_strategy_comparison(output_path=args.output_dir)
    elif args.plot_type == 'stats':
        analyzer.generate_summary_statistics(
            output_path=Path(args.output_dir) / "summary_statistics.csv"
        )
    
    plt.show()


if __name__ == '__main__':
    main()
