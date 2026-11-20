"""
`dqm-driver` with the crv_ml evaluator registered

    python -m mu2e_tranad.evaluator.dqm_driver crv-tranad.yaml [--check] [--commit | --no-commit]

dqm.evaluate.config keeps a fixed registry of the evaluators in the DQM repo, and both
load_config and build_evaluator look the algo up there. This adds crv_ml to it and then
runs the DQM driver's own command line unchanged, so the ML code can live outside the
DQM repo.
"""

import sys


def main(argv=None) -> int:
    from dqm.evaluate import cli, config
    from .ml_evaluator import MLEvaluator

    builtin = config.REGISTRY
    config.REGISTRY = lambda: {**builtin(), MLEvaluator.algo: MLEvaluator}
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
