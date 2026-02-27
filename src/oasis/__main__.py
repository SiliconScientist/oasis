from catbench.adsorption import AdsorptionAnalysis

from oasis.config import get_config
from oasis.subset_mlips import run as analysis_run


def main() -> None:
    cfg = get_config()

    analysis_run(
        base_dir=cfg.analysis.base_dir,
        out_dir=cfg.analysis.out_dir,
        prefixes=cfg.analysis.prefixes,
    )

    config = {
        "mlip_list": [
            "7net-omni",
            "mace-mh-1",
            "mattersim-v1-5m",
            "orb-v3-conservative-inf-omat",
            "uma-s-1p1",
        ],
        "calculating_path": f"{cfg.analysis.out_dir}/ss",
        # "exclude_adsorbates": [],
        "target_adsorbates": ["CCHOH", "OH", "O"],
    }

    analysis = AdsorptionAnalysis(**config)
    analysis.analysis()


if __name__ == "__main__":
    main()
