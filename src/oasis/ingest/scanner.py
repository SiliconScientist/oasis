from pathlib import Path

from oasis.config import get_config


def scan_raw(root: Path):
    entries = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        entries.append(path.name)
    return entries


if __name__ == "__main__":
    cfg = get_config()
    entries = scan_raw(cfg.ingest.source)

    print(f"Found {len(entries)} entries\n")
    for e in entries[:10]:
        print(e)
