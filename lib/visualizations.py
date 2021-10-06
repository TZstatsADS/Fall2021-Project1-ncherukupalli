"""
Filename: visualizations.py
Author: Nikhil Cherukupalli
"""

from matplotlib import pyplot as plt
from wordcloud import WordCloud


def display_wordcloud(data, title=None, save_fig_path=None):
    """ Displays wordcloud highlighting words appearing frequently.
    :param data (list-like): Contains strings of words/sentences.
    :param title (str): Title of wordcloud
    :param save_fig_path (str): Path to save file
    """
    # Build figure
    cloud = WordCloud(background_color='white',
                      max_words=200,
                      max_font_size=40,
                      scale=3,
                      random_state=42
                      ).generate(str(data))
    fig = plt.figure(1, figsize=(10, 10))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(cloud)

    plt.show()
    if save_fig_path:
        fig.savefig(save_fig_path, bbox_inches="tight")


def display_elbow_chart(lst_inertia, save_fig_path=None):
    """ Displays line chart of model inertias. Useful for KMeans to find
        the optimal k.
    :param lst_inertia ([float]): Model inertia results
    :param save_fig_path (str): Path to save file
    :return:
    """
    # Build figure
    fig = plt.figure(1, figsize=(10, 10))
    plt.plot(list(range(1, len(lst_inertia) + 1)),
             lst_inertia,
             "kx-")

    plt.xlabel("# of Clusters")
    plt.ylabel("Inertia")
    plt.grid()

    plt.show()
    if save_fig_path:
        fig.savefig(save_fig_path, bbox_inches="tight")
