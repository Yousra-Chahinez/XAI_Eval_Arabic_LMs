import numpy as np
import random
from datetime import timedelta
import evaluate
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from IPython.display import display, HTML

SEED = 42

def set_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # CPU vars
    torch.manual_seed(seed_value)  # CPU vars
    random.seed(seed_value)  # Python random seed
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # GPU vars
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def compute_metrics(eval_pred):
    metric_name = "accuracy"
    metric = evaluate.load(metric_name)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def format_duration(total_time):
    time_delta = timedelta(seconds=total_time)
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{} hours, {} minutes, {} seconds".format(hours, minutes, seconds)


def hlstr(string, color='white', font_size='21px', font_family='Times New Roman, Times, serif'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style='background-color:{color}; font-size:{font_size}; font-family:{font_family}'>{string}</mark> "

def colorize(attrs, cmap='seismic'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """

    cmap_bound = np.max(np.abs(attrs))
    norm = plt.Normalize(vmin=-1, vmax=1)
    cmap = plt.colormaps[cmap]

    # now compute hex values of colors
    colors = [rgb2hex(cmap(norm(attr))) for attr in attrs]

    return colors

def visualize(tokens, explanation, cmap='seismic'):
    """
    Generate HTML with highlighted words based on input gradients.
    """

    # Colorize the input gradients
    colors = colorize(explanation, cmap)
    highlighted_text = " ".join([hlstr(token, color) for token, color in zip(tokens, colors)])

    # Display the HTML
    display(HTML(highlighted_text))