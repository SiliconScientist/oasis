import argparse

from oasis.mlip.submit import submit_jobs
from oasis.mlip.runner import run_one_task
from oasis.mlip.tasks import make_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="oasis mlip",
        description="Run MLIP processing pipelines",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    # ---- submit ----
    submit_p = subparsers.add_parser(
        "submit", help="Create tasks and submit Slurm job array"
    )
    submit_p.add_argument("--config", default="config.toml")
    submit_p.add_argument("--run-tag", default=None)
    submit_p.add_argument("datasets", nargs="*", help="Optional dataset paths")
    # ---- run-one ----
    run_one_p = subparsers.add_parser(
        "run-one", help="Run a single MLIP task (used inside Slurm)"
    )
    run_one_p.add_argument("--line", required=True)
    run_one_p.add_argument("--config", default="config.toml")
    # ---- make-tasks ----
    make_tasks_p = subparsers.add_parser("make-tasks", help="Generate MLIP task file")
    make_tasks_p.add_argument("--config", default="config.toml")
    make_tasks_p.add_argument("--run-tag", required=True)
    make_tasks_p.add_argument("--out", required=True)
    make_tasks_p.add_argument("datasets", nargs="*")
    args = parser.parse_args(argv)
    if args.subcommand == "submit":
        submit_jobs(
            config_path=args.config,
            run_tag=args.run_tag,
            datasets=args.datasets,
        )
    elif args.subcommand == "run-one":
        run_one_task(
            line=args.line,
            config_path=args.config,
        )

    elif args.subcommand == "make-tasks":
        make_tasks(
            config_path=args.config,
            run_tag=args.run_tag,
            out_path=args.out,
            datasets=args.datasets,
        )
