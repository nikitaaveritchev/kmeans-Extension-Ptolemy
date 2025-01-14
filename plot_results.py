import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot(
    input_file="results/clustering_performance_results_formatted.xlsx",
    output_file="results/combined_plot.pdf",
):
    data = pd.ExcelFile(input_file)
    df = pd.concat(
        [
            data.parse("Results1"),
            data.parse("Results2"),
        ]
    )

    df.Dataset = df.Dataset.str.replace("Random", "Uniform")
    df = df[df["K"].isin([3, 20, 100])]
    df = df[~df.Dataset.isin(["Circles", "Moons"])]

    # Function to calculate speedup
    metric = "Distance_Evaluations"

    def calculate_speedup(group):
        reference = group[group["Method"] == "Elkan"][metric].values[0]
        group["Speedup"] = reference / group[metric]
        return group

    all_speedups = (
        df.groupby(["Dataset", "K"])[df.columns]
        .apply(calculate_speedup)
        .reset_index(drop=True)
    )
    speedups = all_speedups[
        all_speedups.Method.isin(["Ptolemy_upper", "Ptolemy_lower", "Ptolemy"])
    ]

    fontscale = 1
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.size": 10 * fontscale,
            "axes.titlesize": 10 * fontscale,
            "axes.labelsize": 10 * fontscale,
            "legend.fontsize": 10 * fontscale,
            "xtick.labelsize": 10 * fontscale,
            "ytick.labelsize": 10 * fontscale,
        }
    )

    sns.set_palette("colorblind")

    # width of available page is ~5 inches, we don't have a lot of height left, so only allocating 3 inches for height
    scale = 1.7
    fig = plt.figure(figsize=(5 * scale, 2.5 * scale), dpi=300, layout="constrained")
    # fig = plt.figure(layout="constrained")

    layout = [
        ["Iris", "Wine", "legend"],
        ["Gaussian_low", "Gaussian_medium", "Gaussian_high"],
        ["Uniform_low", "Uniform_medium", "Uniform_high"],
    ]
    ax_mosaic = fig.subplot_mosaic(
        layout,
        sharex=True,
    )

    # sharey does not work as intended, so we have to set lables manually
    for col in layout:
        first = col[0]
        for entry in col[1:]:
            if entry == "empty":
                continue
            ax_mosaic[entry].set_ylabel("")
            ax_mosaic[entry].tick_params(labelleft=False)
            ax_mosaic[entry].sharey(ax_mosaic[first])

    def plot_single(data, name, legend=False):
        axis = ax_mosaic[name]

        axis.axhline(y=1, color="black", linestyle="-", linewidth=1, alpha=0.8)

        data = data.copy()
        data["K"] = data["K"].astype("str")  # label shoud be categorical

        plt_kwargs = dict(
            x="K",
            y="Speedup",
            hue="Method",
            legend=legend,
            style="Method",
            markers=True,
        )
        sns.lineplot(
            data=data,
            ax=axis,
            **plt_kwargs,
        )
        axis.grid(visible=True, axis="y")
        axis.set_xlabel("$k$")
        axis.set_title(name.replace("_", ", "))

    for name, dataset in speedups.groupby("Dataset"):
        if name == "Wine":
            plot_single(dataset, name, legend=True)
        else:
            plot_single(dataset, name)

    # generate legend inside the "fake" plot
    ax_legend = ax_mosaic["legend"]

    # copy over legend
    handles, labels = ax_mosaic["Wine"].get_legend_handles_labels()
    ax_mosaic["Wine"].get_legend().remove()
    ax_legend.legend(handles, labels, loc="center")
    ax_legend.axis("off")

    for col in layout:
        axis = ax_mosaic[col[0]]
        yticks = axis.get_yticks().tolist()
        if 1 not in yticks:
            yticks.append(1)
        yticks.sort()
        axis.set_yticks(yticks)

    plt.savefig(output_file, format="pdf")


if __name__ == "__main__":
    print("Producing plot")
    plot()
