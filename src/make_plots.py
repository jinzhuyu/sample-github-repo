import matplotlib.pyplot as plt


def create_parity_plot(data, path):
    observed = data["observed_peak_flow_cms"]
    predicted = data["predicted_peak_flow_cms"]

    plt.figure(figsize=(5, 4))
    plt.scatter(observed, predicted, alpha=0.75)
    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Observed peak flow (cms)")
    plt.ylabel("Predicted peak flow (cms)")
    plt.title("Parity plot: observed vs predicted peak flow")
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def create_residual_plot(data, path):
    predicted = data["predicted_peak_flow_cms"]
    residual = data["observed_peak_flow_cms"] - data["predicted_peak_flow_cms"]

    plt.figure(figsize=(5, 4))
    plt.scatter(predicted, residual, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted peak flow (cms)")
    plt.ylabel("Residual (observed - predicted)")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()