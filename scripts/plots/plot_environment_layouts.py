from rl_training.maps import maps
from matplotlib import pyplot as plt
import matplotlib as mpl


def run():
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 2))

    plot_timer_treasure(axes[0])
    plot_moving_treasure(axes[1])
    plot_key_treasure(axes[2])
    plot_charger_treasure(axes[3])
    plt.tight_layout()

    fig_path = "out/fig/env_layouts.png"
    fig.savefig(fig_path, format='png', dpi=300)

    plt.show()


def plot_timer_treasure(axis):
    data = maps['timertreasure']['boxes']
    add_bounding_box(axis, data['spawn_left']['coords'], data['spawn_left']['face_colour'])
    add_bounding_box(axis, data['spawn_right']['coords'], data['spawn_right']['face_colour'])
    add_bounding_box(axis, data['treasure']['coords'], data['treasure']['face_colour'])
    axis.set_title('Timer')
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect('equal')


def plot_moving_treasure(axis):
    data = maps['movingtreasure']['boxes']
    add_bounding_box(axis, data['spawn_left']['coords'], data['spawn_left']['face_colour'])
    add_bounding_box(axis, data['spawn_right']['coords'], data['spawn_right']['face_colour'])
    add_bounding_box(axis, data['treasure']['coords'], data['treasure']['face_colour'])
    axis.arrow(0.35, 0.8, -0.15, 0.0, head_width=0.05, color='k')
    axis.arrow(0.65, 0.8, 0.15, 0.0, head_width=0.05, color='k')
    axis.set_title('Moving')
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect('equal')


def plot_key_treasure(axis):
    data = maps['keytreasure_A']['boxes']
    add_bounding_box(axis, data['spawn_left']['coords'], data['spawn_left']['face_colour'])
    add_bounding_box(axis, data['spawn_right']['coords'], data['spawn_right']['face_colour'])
    add_bounding_box(axis, data['treasure']['coords'], data['treasure']['face_colour'])
    add_bounding_box(axis, data['key']['coords'], data['key']['face_colour'])
    axis.set_title('Key')
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect('equal')


def plot_charger_treasure(axis):
    data = maps['chargertreasure']['boxes']
    add_bounding_box(axis, data['spawn_left']['coords'], data['spawn_left']['face_colour'])
    add_bounding_box(axis, data['spawn_right']['coords'], data['spawn_right']['face_colour'])
    add_bounding_box(axis, data['treasure']['coords'], data['treasure']['face_colour'])
    add_bounding_box(axis, data['charge_zone']['coords'], 'darkgoldenrod')
    axis.set_title('Charger')
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect('equal')


def add_bounding_box(axis, bounds, color, lw=1, alpha=0.5):
    x0, y0, x1, y1 = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
    rect = mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha)
    axis.add_patch(rect)
    return rect


if __name__ == "__main__":
    run()
