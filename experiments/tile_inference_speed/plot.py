"""
===========================
Plots with different scales
===========================

Demonstrate how to do two plots on the same axes with different left and
right scales.

The trick is to use *two different axes* that share the same *x* axis.
You can use separate `matplotlib.ticker` formatters and locators as
desired since the two axes are independent.

Such axes are generated by calling the `Axes.twinx` method.  Likewise,
`Axes.twiny` is available to generate axes that share a *y* axis but
have different top and bottom scales.

The twinx and twiny methods are also exposed as pyplot functions.

"""

import numpy as np
import matplotlib as mpl
mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                   # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
    "font.size": 22
}
mpl.rcParams.update(pgf_with_rc_fonts)
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

resolutions = np.array(['3584x1792', '1792x1792', '896x896', '448x448', '224x224'])
accuracy = np.array([0.574, 0.719, 0.893, 0.937, 0.964])
processing_time = np.array([19.0, 29.0, 89.0, 328.0, 1280.0])
fps = 1000.0 / processing_time

fig, ax1 = plt.subplots()
t = 1 + np.arange(len(resolutions))
ax1.plot(t, accuracy, 'b-', t, accuracy, 'bs')
ax1.set_xlabel('Tile Resolution')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Accuracy', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim([0, 1.0])
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xticklabels(resolutions)

ax2 = ax1.twinx()
ax2.plot(t[:len(fps)], fps, 'r-', t[:len(fps)], fps, 'ro')
ax2.set_ylabel('Speed (FPS) on Jetson TX2', color='r')
ax2.tick_params('y', colors='r')

# fig.tight_layout()
plt.savefig('tile_resolution_vs_speed_accuracy.pdf', bbox_inches='tight')
