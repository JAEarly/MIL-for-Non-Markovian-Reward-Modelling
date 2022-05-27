"""
Plotting library
"""

from matplotlib import rcParams
from matplotlib.pyplot import show as plt_show

from .ax import single_axis as ax
from .templates import *

def font_size(size): rcParams.update({"font.size": size, "axes.labelpad": size})
def line_width(lw): rcParams["lines.linewidth"] = lw

font_size(5.5)
line_width(0.693)
rcParams.update({
    "font.family":        "serif",
    "font.serif":         "ETBembo",
    "svg.fonttype":       "none",    # Makes text editable and reduces file size
    "axes.linewidth":     0.369,
    "lines.markersize":   0.3,
    "axes.unicode_minus": False      # Prevents "RuntimeWarning: Glyph 8722 missing from current font."
})

for typ in ("major","minor"):
    rcParams[f"xtick.{typ}.width"] = 0.369
    rcParams[f"xtick.{typ}.size"] =  1
    rcParams[f"xtick.{typ}.pad"] =   1
    rcParams[f"ytick.{typ}.width"] = 0.369
    rcParams[f"ytick.{typ}.size"] =  1
    rcParams[f"ytick.{typ}.pad"] =   0

def show(): plt_show()
