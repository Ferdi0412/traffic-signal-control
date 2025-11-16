import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    """To be used for comparison of traffic matrics between DQN and SCATS"""
    
    def __init__(self, figsize=(12, 6), style='default'):
        """
        Initialize the plotter
        """
        self.figsize = figsize
        plt.style.use(style)
    
  
    def plot_metrics_comparison(self, data_dict, color1='steelblue', color2='coral',
                        show_values=True, decimal_places=2):
        """
        Plot metrics of 2 inputs for comparison
        
        Args:
            data_dict: Dictionary with structure:
                      {
                          'data1': DQNavg_qlength,
                          'data2': SCATSavg_qlength,
                          'barlabels': [f"{directions_12x1[i]}" for i in range(len(DQN_qlength_flat))],
                          'label1': 'DQN',
                          'label2': 'SCATS',
                          'xlabel': 'NSEW',
                          'ylabel': 'Queue Length'
                          'title': 'Queue Length Comparison Between DQN and SCATS',
                          'save_dir': True
                      }
            color1: Color for first dataset
            color2: Color for second dataset
            show_values: Whether to show values on bars
            decimal_places: Number of decimal places for values
        """
        # Extract data from dictionary
        data1 = data_dict['data1']
        data2 = data_dict['data2']
        barlabel = data_dict['barlabel']
        label1 = data_dict.get('label1')
        label2 = data_dict.get('label2')
        xlabel = data_dict.get('xlabel')
        ylabel = data_dict.get('ylabel')
        title = data_dict.get('title')
        save_dir = data_dict['save_dir']
        # Flatten arrays if needed
        if isinstance(data1, np.ndarray) and data1.ndim > 1:
            data1 = data1.flatten()
        if isinstance(data2, np.ndarray) and data2.ndim > 1:
            data2 = data2.flatten()
        
        # Set up bar positions
        x = np.arange(len(data1))
        width = 0.35
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        bars1 = ax.bar(x - width/2, data1, width, label=label1, color=color1)
        bars2 = ax.bar(x + width/2, data2, width, label=label2, color=color2)
        
        # Add labels and title
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(barlabel)
        ax.legend()
        
        # Add value labels on bars
        if show_values:
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.{decimal_places}f}',
                           ha='center', va='bottom', fontsize=8)
        
        # Styling
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        # Save or show
        if save_dir:
            plt.savefig(title, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.show()
        
        plt.close()
    
    def plot_rewards_comparison(self, data_dict, 
                     title='Multiple Array Comparison',
                     xlabel='Items', ylabel='Values',
                     colors=None, save_path=None, 
                     show_values=True, decimal_places=2):
        """
        Plot rewards comparison
        
        Args:
            data_dict: Dictionary with structure:
                      {
                          'scenarios': ['DQNAgent', 'SCATS'],
                          'plt_rewards': [DQNcompare_reward, SCATScompare_reward],
                          'plt_deltaq': [DQNcompare_deltaq, SCATScompare_deltaq],
                          'plt_longwait': [DQNcompare_longwait, SCATScompare_longwait],
                          'xlabel': 'DQN vs SCATS',
                          'ylabel': 'Reward Components',
                          'title': 'Queue Length Comparison Between DQN and SCATS',
                          'save_dir': True
                      }
            colors: List of colors for each dataset
            show_values: Whether to show values on bars
            decimal_places: Number of decimal places
        """
        # Extract data from dictionary
        scenarios = data_dict.get('scenarios')
        plt_rewards = data_dict['plt_rewards']
        plt_deltaq = data_dict['plt_deltaq']
        plt_longwait = data_dict['plt_longwait']
        xlabel = data_dict.get('xlabel')
        ylabel = data_dict.get('ylabel')
        title = data_dict.get('title')
        save_dir = data_dict['save_dir']

        # Set up bar positions
        x = np.arange(len(scenarios))  # Label locations
        width = 0.25  # Width of bars

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bars
        bars1 = ax.bar(x - width, plt_rewards, width, label='Total Rewards', color='steelblue')
        bars2 = ax.bar(x, plt_deltaq, width, label='Delta Qlength', color='coral')
        bars3 = ax.bar(x + width, plt_longwait, width, label='Penalty for Long Wait Time', color='mediumseagreen')

        # Add labels and title
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend()

        # Add value labels on top of bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='top', fontsize=9)

        # Add grid for easier reading
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        
        # Save or show
        if save_dir:
            plt.savefig(title, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.show()
        
        plt.close()
