import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def get_section_results(file):
    """
    requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == "Train_EnvstepsSoFar":
                X.append(v.simple_value)
            elif v.tag == "Eval_AverageReturn":
                Y.append(v.simple_value)
    return X, Y


if __name__ == "__main__":
    sns.set_theme("paper", "whitegrid")
    c = sns.color_palette()
    cred = sns.color_palette("rocket")
    fig = plt.figure(figsize=(6, 3), layout="constrained")

    logdirs = [
        "data_final_reacher/reacher_random_nn_reacher_05-12-2023_20-06-42/events*",
        "data_final_reacher/reacher_random_gp_reacher_05-12-2023_22-01-21/events*",
        "data_final_reacher/reacher_random_explore_1_reacher_05-12-2023_23-31-03/events*",
        "data_final_reacher/reacher_random_explore_3_reacher_06-12-2023_01-13-35/events*",
        "data_final_reacher/reacher_random_explore_5_reacher_06-12-2023_03-04-38/events*",
    ]
    names = ["NN", "GP", "GP + Explore 1", "GP + Explore 3", "GP + Explore 5"]
    colors = [c[0], cred[3], cred[2], cred[1], cred[0]]

    for n, l, c in zip(names, logdirs, colors):
        eventfile = glob.glob(l)[0]

        X, Y = get_section_results(eventfile)
        plt.plot(Y, label=n, color=c)

    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    plt.legend()
    plt.title("7DoF Reacher")
    plt.savefig("results/reacher_eval.png", dpi=300)
    # plt.show()
