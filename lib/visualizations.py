"""
Filename: visualizations.py
Author: Nikhil Cherukupalli
"""

from wordcloud import WordCloud
from matplotlib import pyplot as plt


def display_wordcloud(data, title=None, save_fig_path=None):
    cloud = WordCloud(
        background_color='white',
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
    fig, axs = plt.figure(figsize=(16, 9))

    axs.plot(list(range(1, len(lst_inertia) + 1)), lst_inertia, "kx-")
    axs.set_xlabel("# of Clusters")
    axs.set_ylabel("Inertia")

    plt.show()
    if save_fig_path:
        fig.savefig(save_fig_path, bbox_inches="tight")
