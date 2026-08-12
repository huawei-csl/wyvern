import os
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

def set_plot_params():
    custom_params = {"axes.spines.right": True, "axes.spines.top": True}
    sns.set_theme(style="ticks", rc=custom_params, font_scale=1)
    plt.style.use(['science', 'grid', 'nature', 'std-colors', "no-latex"])


def plot_length_resources_report(values: Iterable[int], 
                                 topic: str, 
                                 plot_folder: str, 
                                 plot_name: str):
    """Create a barplot that displays the token count for the resources, the
    summaries and the report.
    
    Parameters
    ----------
    - values [`Iterable[int]`]: token counts of the resources, the summaries and
    the report (thus, its length should be three)
    - topic[`str`]: topic
    - plot_folder [`str`]: folder where to save the plot
    - plot_name[`str`]: name of the plot file
    """
    set_plot_params()
    fig, ax = plt.subplots(1, 1, figsize = (3,2))

    labels = ['Resources', 'Summaries', "Report"] 
    colors = ["#007EA7", "#007EA7", "#AD2E24"] 

    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.25)

    bars[1].set_hatch('//')  
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05 * values[1],
                f'{yval}', ha='center', va='bottom', fontsize=6)
        
    ax.set_ylim(0, values[0] + 0.25 * values[0])
    ax.set_ylabel('Tokens count')
    ax.set_title("Topic: " + f"\\emph{{{topic}}}")
    os.makedirs(plot_folder, exist_ok=True)
    fig.savefig(os.path.join(plot_folder, 
                            plot_name),
                            format="png",
                            dpi=900,
                            transparent=True)
    

def plot_recall_stats(supported: int, 
                      no_cit: int, 
                      ref_not_found: int, 
                      unsupported_claim: int,
                      plot_folder: str, 
                      plot_name: str):
    """Plot the results related to the citation recall computation.
    
    Parameters
    ----------
    - supported [`int`]: number of supported claims
    - no_cit [`int`]: number of claims without any reference
    - ref_not_found [`int`]: number of claims for which at least one reference
    was not found in the DB
    - unsupported_claim [`int`]: number of logically unsupported claims
    - plot_folder [`str`]: folder where to save the plot
    - plot_name[`str`]: name of the plot file
    """
    set_plot_params()

    # Stacked bar heights
    total = no_cit + ref_not_found + unsupported_claim
    if total != 0:
        percentage_unsupported_claim = unsupported_claim / total * 100
        percentage_ref_not_found = ref_not_found / total * 100
        percentage_no_cit = no_cit / total * 100
    else:
        percentage_unsupported_claim = 0.
        percentage_ref_not_found = 0.
        percentage_no_cit = 0.

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4,2))

    # Left bar (single bar)
    ax.bar(1, supported, width=0.3, color='#1985A1',edgecolor='black', linewidth=0.25)

    # Right bar (stacked bar)
    ax.bar(1.5, unsupported_claim, width=0.3, label=f'No logical support ({percentage_unsupported_claim:.1f}\%)', color='#F5CB5C',edgecolor='black', linewidth=0.25)
    ax.bar(1.5, ref_not_found, bottom=unsupported_claim, width=0.3, label=f'Reference not found ({percentage_ref_not_found:.1f}\%)', color='#E88873',edgecolor='black', linewidth=0.25)
    ax.bar(1.5, no_cit, bottom=unsupported_claim+ref_not_found, width=0.3, label=f'No citations ({percentage_no_cit:.1f}\%)', color='#A62639',edgecolor='black', linewidth=0.25)

    # Add labels
    ax.set_ylabel('Count')

    # Set x-ticks to show the bars correctly
    ax.set_xticks([1, 1.5])
    ax.set_xticklabels(['Supported', 'Not supported'])

    # Add legend to the right of the right bar
    ax.legend(title="Cause", loc='upper left', bbox_to_anchor=(1.05, 1))

    # Show plot with tight layout
    plt.tight_layout()
    os.makedirs(plot_folder, exist_ok=True)
    fig.savefig(os.path.join(plot_folder, 
                             plot_name),
                             format="png",
                             dpi=900,
                             transparent=True)
    

def plot_logits_probs(logits_of_interest, 
                      temperature, 
                      filename, 
                      tokens_of_interest=["1", "2", "3", "4", "5"]):
    set_plot_params()
    import torch
    
    probs = torch.nn.functional.softmax(torch.tensor(logits_of_interest) / temperature, dim=-1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))
    
    ax1.bar(tokens_of_interest, logits_of_interest, color=sns.color_palette()[3], linewidth=0.4, edgecolor="black")
    ax1.set_xlabel('Tokens')
    ax1.set_ylabel('Value')
    ax1.set_title('Logits')

    for i, p in enumerate(logits_of_interest):
        ax1.text(i, p + 0.01, f"{p:.2f}", ha='center')

    ax2.bar(tokens_of_interest, probs, color=sns.color_palette()[4], linewidth=0.4, edgecolor="black")
    ax2.set_xlabel('Tokens')
    ax2.set_ylabel('Value')
    ax2.set_title('Probability (w/ temperature)')
    ax1.set_ylim(0,max(logits_of_interest) + 0.1*max(logits_of_interest))
    ax2.set_ylim(0,max(probs) + 0.1*max(probs))

    # Add annotations
    for i, p in enumerate(probs):
        ax2.text(i, p + 0.01, f"{p:.2f}", ha='center')

    fig.suptitle(f"Temperature {temperature}", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{filename}.png", format='png', dpi=900, transparent=True)
    plt.close()