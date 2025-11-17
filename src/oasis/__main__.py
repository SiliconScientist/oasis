from oasis.config import get_config
from oasis.processing import get_data


def main():
    cfg = get_config()
    df = get_data(cfg)
    print(df)


if __name__ == "__main__":
    main()
