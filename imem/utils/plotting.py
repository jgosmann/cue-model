import os
import os.path


def save_tight_fig(fig, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.)
