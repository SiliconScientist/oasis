import torch
import polars as pl

from oasis.config import get_config
from oasis.processing import get_loaders
from oasis.model import get_model
from oasis.plot import plot_predictions


def main():
    cfg = get_config()
    loaders = get_loaders(cfg)
    model = get_model(cfg)
    total_loss = 0.0
    predictions = []
    for batch in loaders.test:
        x, y, atomic_reference = batch
        x = {k: v.to(cfg.trainer.device) for k, v in x.items()}
        y = y.to(cfg.trainer.device)
        atomic_reference = atomic_reference.to(cfg.trainer.device)
        slab_pred = model(x["slab"])
        ads_slab_pred = model(x["ads_slab"])
        ads_energy = ads_slab_pred - (slab_pred + atomic_reference)
        loss = torch.mean(torch.abs(ads_energy - y))
        total_loss += loss.item()
        preds = [
            {"y_pred": m.item(), "y_true": label.item()}
            for m, label in zip(ads_energy, y)
        ]
        predictions.extend(preds)
    print(total_loss / len(loaders.test))
    df = pl.DataFrame(predictions)
    df.write_parquet("results/predictions.parquet")
    df = pl.read_parquet(source="results/predictions.parquet")
    plot_predictions(df=df)


if __name__ == "__main__":
    main()
