import torch

from oasis.config import get_config
from oasis.processing import get_loaders
from oasis.model import get_model


def main():
    cfg = get_config()
    loaders = get_loaders(cfg)
    model = get_model(cfg)
    loss = 0.0
    for batch in loaders.test:
        x, y = batch
        sys_pred = model(x["system"])
        surf_pred = model(x["surface"])
        ads_energy = sys_pred - (surf_pred - 10.681)
        loss += torch.mean(torch.abs(ads_energy - y))
    print(loss / len(loaders.test))


if __name__ == "__main__":
    main()
