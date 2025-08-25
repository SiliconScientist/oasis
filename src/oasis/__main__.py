from oasis.config import get_config
from oasis.processing import get_loaders


def main():
    cfg = get_config()
    loaders = get_loaders(cfg)
    print(len(loaders.holdout.dataset), len(loaders.test.dataset))


if __name__ == "__main__":
    main()
