from oasis.config import get_config


def main():
    cfg = get_config()
    print(f"Seed from config: {cfg.seed}")


if __name__ == "__main__":
    main()
