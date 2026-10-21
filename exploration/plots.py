import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

SAVE_PATH = Path("results", "graphs")


def plot(df: pd.DataFrame, type: int, save: bool, show=True):
    stress_labels = {0: "Low", 1: "Medium", 2: "High"}
    match type:
        case 0:  # barplot of stress level
            stress_levels = df["stress_level"].value_counts().sort_index()
            plt.bar(
                x=stress_levels.index.map(stress_labels),
                height=stress_levels.values,
                edgecolor="black",
            )
            plt.title("Distribution of Stress Level")
            plt.xlabel("Stress level")
            plt.ylabel("Number of students")

            if save:
                plt.savefig(Path(SAVE_PATH, "Barplot of Stress Levels"))

        case 1:  # sleep_quality vs stress level
            sns.stripplot(
                data=df,
                x=df["stress_level"].map(stress_labels),
                y="sleep_quality",
                jitter=0.05,  # Adds random noise to reduce point overlap
                alpha=0.5,  # Adds transparency to visualize density
                size=7,
                hue=df["stress_level"].map(stress_labels),
                order=["Low", "Medium", "High"],
            )
            plt.title("Sleep Quality vs Stress Level")
            plt.xlabel("Stress Level")
            plt.ylabel("Sleep Quality")

            if save:
                plt.savefig(Path(SAVE_PATH, "Sleep Quality vs Stress Level"))

        case 2:
            sns.stripplot(
                data=df,
                x=df["stress_level"].map(stress_labels),
                y="academic_performance",
                jitter=0.05,  # Adds random noise to reduce point overlap
                alpha=0.5,  # Adds transparency to visualize density
                size=7,
                hue=df["stress_level"].map(stress_labels),
                order=["Low", "Medium", "High"],
            )
            plt.title("Stress Level vs Academic Performance")
            plt.xlabel("Stress Level")
            plt.ylabel("Academic Performance")

            if save:
                plt.savefig(Path(SAVE_PATH, "Academic Performance vs Stress Level"))
        case _:
            pass

    if show:
        plt.show()
    else:
        plt.clf()
