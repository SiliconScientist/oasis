from oasis.config import get_config
from oasis.processing import get_data


def main():
    cfg = get_config()
    relaxed_slabs, relaxed_ads_slabs, y_labels = get_data(cfg)


if __name__ == "__main__":
    main()
