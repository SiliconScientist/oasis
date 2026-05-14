import matplotlib.pyplot as plt


def plot_parity(y_test, y_pred, rmse, mae, output_path="parity_plot.png"):
    """
    Create a parity plot comparing predicted vs actual values.

    Parameters:
    -----------
    y_test : array-like
        Actual target values
    y_pred : array-like
        Predicted values
    rmse : float
        Root mean square error
    mae : float
        Mean absolute error
    output_path : str
        Path to save the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    lim = [min(y_test.min(), y_pred.min()) - 0.1, max(y_test.max(), y_pred.max()) + 0.1]
    _, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlabel("DFT adsorption energy (eV)")
    ax.set_ylabel("Predicted adsorption energy (eV)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title(f"RMSE={rmse:.3f} eV  MAE={mae:.3f} eV")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    return ax.figure
