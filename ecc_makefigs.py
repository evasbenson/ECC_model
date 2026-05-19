"""
translated from ECC_makefigs.m
creates the utility heatmap visualization from saved simulation results.
uses matplotlib's pcolormesh which is the equivalent of matlab's pcolor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import glob


def create_heatmap(filename=None):
    """
    loads simulation results and creates the heatmap plot.
    same visualization logic as the matlab script: jet colormap with
    NaN values shown as black (infeasible) and Inf values as white (solver failure).

    args:
        filename: path to .npz results file. if None, uses the most recent one.
    """

    # find the results file
    if filename is None:
        files = glob.glob("ECC_solutions_*.npz")
        if not files:
            print("no results files found. run ecc_opt_wrapper.py first.")
            return
        filename = max(files)
        print(f"loading most recent results: {filename}")

    # load results (equivalent to loading the .mat file in matlab)
    data = np.load(filename, allow_pickle=True)
    utilmat = data['utilmat']
    tau = data['tau']
    pop = data['pop']

    # prepare plot data (same logic as matlab script)
    plotdata = utilmat.copy()

    # get the actual min and max of finite values
    finite_mask = np.isfinite(plotdata)
    if not np.any(finite_mask):
        print("no finite values in utility matrix, nothing to plot")
        return

    actual_min = np.min(plotdata[finite_mask])
    actual_max = np.max(plotdata[finite_mask])

    # replace -Inf (solver failures) with a value just below the minimum
    # same as matlab: plotdata(isinf(plotdata)) = actual_min - 1
    plotdata[np.isinf(plotdata)] = actual_min - 1

    # create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # build custom colormap: white for replaced Inf values, then jet for real data
    # this matches the matlab code's customMap = [white; standardMap]
    jet = plt.cm.jet
    colors = [[1, 1, 1]]  # white for solver failures
    colors.extend(jet(np.linspace(0, 1, 256)).tolist())
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    # create meshgrid and plot (pcolormesh = matlab's pcolor)
    Pop, Tau = np.meshgrid(pop, tau)
    im = ax.pcolormesh(Pop, Tau, plotdata, cmap=custom_cmap, shading='auto')

    # set color limits (same as matlab's caxis)
    im.set_clim(actual_min - 1, actual_max)

    # black background for NaN values (same as matlab: set(gca, 'Color', 'k'))
    ax.set_facecolor('black')

    # colorbar
    plt.colorbar(im, ax=ax)

    # labels and title (same as matlab)
    ax.set_xlabel('Population (billions)')
    ax.set_ylabel('Technology Level (τ)')
    ax.set_title('Utility Heatmap')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # save figure
    output_filename = filename.replace('.npz', '_heatmap.png')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"figure saved to {output_filename}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        create_heatmap(sys.argv[1])
    else:
        create_heatmap()
