import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA = Path("data")


def plot_cartpole_performance():
    sns.set_theme("paper", "whitegrid")

    data = pd.DataFrame()
    for p in DATA.glob("*"):
        if "inverted-pendulum" in str(p):
            new = pd.read_csv(p / "data.csv")

            if "gp" in str(p):
                explore = int(str(p).split("explore")[1].split("_")[0]) + 1
                if explore == 0:
                    new["Model"] = "GP"
                else:
                    new["Model"] = f"GP-E{explore}"

            else:
                new["Model"] = "NN"

            data = pd.concat([data, new])

    data["itr"] += 1
    data.rename(
        columns={"Eval_AverageEpLen": "Average Episode Length", "itr": "Iterations"},
        inplace=True,
    )

    fig = plt.figure(figsize=(8, 4), layout="constrained")
    sns.lineplot(
        data=data,
        x="Iterations",
        y="Average Episode Length",
        hue="Model",
        hue_order=["GP", "GP-E5", "GP-E10", "GP-E15", "NN"],
        err_kws={"alpha": 0.2},
    )
    plt.savefig("invpend_eval.png")


def plot_cartpole_heatmap():
    sns.set_theme("paper", "whitegrid")

    data = {}
    for p in DATA.glob("*"):
        if "inverted-pendulum" in str(p):
            new = np.load(p / "obs.npy")

            if "gp" in str(p):
                explore = int(str(p).split("explore")[1].split("_")[0]) + 1
                if explore == 0:
                    name = "GP"
                else:
                    name = f"GP-E{explore}"

            else:
                name = "NN"

            if name not in data:
                data[name] = new

    fig, axs = plt.subplots(1, 5, figsize=(8, 3), layout="constrained", sharey=True)

    for i, name in enumerate(["GP", "GP-E5", "GP-E10", "GP-E15", "NN"]):
        d = data[name]
        d[:, 1] *= 180 / np.pi
        df = pd.DataFrame(d[:, :2], columns=["X", "Theta"])
        sns.histplot(
            df,
            x="X",
            y="Theta",
            ax=axs[i],
            cbar=False,
            stat="probability",
            bins=[75, 75],
        )
        axs[i].set_title(name)

    plt.savefig("invpend_heatmap.png")


if __name__ == "__main__":
    plot_cartpole_performance()
    plot_cartpole_heatmap()
