import matplotlib.pyplot as plt


def plot_predictions(df, output_path="results/parity.png"):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["y_true"], df["y_pred"], alpha=0.5)
    plt.plot(
        [df["y_true"].min(), df["y_true"].max()],
        [df["y_true"].min(), df["y_true"].max()],
        color="red",
        linestyle="--",
    )
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Model Predictions vs True Values")
    plt.savefig(output_path)
    plt.close()
