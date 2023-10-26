import pyarrow as pa
import numpy as np
import matplotlib.pyplot as plt
from modules.plot_constants import uzh_colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def convert_to_array_from_string(string_array):
    res = string_array.replace("]", "").replace("[", "").replace("\n", "")
    arr = res.split(" ")
    arr = np.array([float(i) for i in arr if i!=""])
    return arr

def get_configuration_subplots_training_loss(df_train_curves, title, file_path=None, df_validation_curves=None, do_ylim = False):
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14, 10))
    time = None

    colorscheme = [uzh_colors["blue"], uzh_colors["red"], uzh_colors["green"]]

    for i, row in df_train_curves.iterrows():
        lr = row["index"]
        label = "lr = {}".format(lr)
        row_color = colorscheme[i]
        
        for j, c in enumerate(df_train_curves.columns[1:]):
            data_train = row[c]
            
            time = range(len(data_train))
            k, l = j // 2, j%2

            # time = time[:50]
            # data = data[:50]
            
            axes[k, l].plot(time, data_train, label=label, color = row_color)
            
            if df_validation_curves is not None:
                data_val = df_validation_curves.loc[i, c]
                axes[k, l].plot(time, data_val, label="_nolegend_", color = row_color, linestyle='--')

    for j, c in enumerate(df_train_curves.columns[1:]):
        k, l = j // 2, j%2
        axes[k,l].set_title("N hidden: "+c)
        axes[k,l].set_xlabel("Epoch")
        ylabel = "MSE training loss"
        if df_validation_curves is not None:
            ylabel = "MSE loss"
        axes[k,l].set_ylabel(ylabel)
        
        axes[k,l].set_xlim(min(time), max(time))

        axes[k,l].grid(zorder=100, lw =0.5, color = 'lightgray')
        leg = axes[k,l].legend(frameon=True,facecolor='white', framealpha=1)
        frame = leg.get_frame()
        frame.set_linewidth(0)
    
    lines = [
        Line2D([], [], linestyle='-', label='Training loss', color='black'),
        Line2D([], [], linestyle='--', label='Validation Loss', color='black')
    ]

    fig.legend(loc='center', bbox_to_anchor=(0.5, -0.02), handles = lines)

    # Adjust layout to prevent subplot overlap
    fig.tight_layout()
    
    if do_ylim:
        plt.ylim((0, 1))
    plt.yticks(fontsize=22)
    plt.suptitle(title)
    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path, bbox_inches = 'tight')

    plt.show()
    