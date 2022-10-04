from matplotlib import rcParams, pyplot as plt


def single_axis( 
    w_inches,     h_inches, 
    off=False,
    x_label=None, y_label=None,
    x_lims=None,  y_lims=None,
    x_ticks=None, y_ticks=None,
    top=False,    right=False
    ): 
    fig, ax = plt.subplots()
    ax.show = plt.show
    ax.save = fig.savefig
    fig.set_size_inches(w_inches, h_inches)
    ax.margins(x=0, y=0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if off: raise NotImplementedError()
    else:
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        if x_ticks is not None: ax.set_xticks(x_ticks)
        if y_ticks is not None: ax.set_yticks(y_ticks)
        if x_lims is not None: ax.set_xlim(*x_lims)
        if y_lims is not None: ax.set_ylim(*y_lims)
        if top:   
            ax.spines["bottom"].set_visible(False); ax.xaxis.tick_top();   ax.xaxis.set_label_position("top")  
            for label in ax.get_xticklabels(): label.set_verticalalignment("baseline")
            ax.xaxis.set_tick_params(pad=rcParams["xtick.major.pad"]+0.7) # NOTE: Annoying offsets
            ax.xaxis.get_label().set_verticalalignment("baseline")
            ax.xaxis.labelpad = rcParams["axes.labelpad"]+1.531
        else:  
            ax.spines["top"].set_visible(False)
        if right: 
            ax.spines["left"].set_visible(False);   ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")  
            for label in ax.get_yticklabels(): label.set_horizontalalignment("left")
            ax.yaxis.set_tick_params(pad=rcParams["xtick.major.pad"])
        else:          
            ax.spines["right"].set_visible(False)
        plt.xticks(ha="center")
        plt.yticks(va="center", rotation=90)
    return ax
