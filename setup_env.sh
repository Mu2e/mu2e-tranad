# Environment recipe for running mu2e_tranad (incl. TranAD) on a CPU Mu2e gpvm
# (e.g. mu2egpvm05).
#
# This repo must be in a directory named mu2e_tranad/ (it is the Python package).
# It no longer needs a CrossExperimentalAIDQM checkout: the TranAD model is
# copied from anldq into training/anl_tranad/.
#
# Usage (from anywhere):
#   source mu2e_tranad/setup_env.sh
#   python -m mu2e_tranad --dqm-source crvreco,kpp_crv,file,005-001 --output results/<dir>
#
# Base: Mu2e cvmfs "ana" env (Python 3.12, torch 2.5.1, pandas, sklearn, matplotlib).
# seaborn (used by analysis/visualization.py) is missing there, so it is installed
# once into mu2e_tranad/.ana_extras and added to PYTHONPATH.

_PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PARENT="$(dirname "$_PKG")"
_EXTRAS="$_PKG/.ana_extras"
_ANA=/cvmfs/mu2e.opensciencegrid.org/env/ana/current

if [ "$(basename "$_PKG")" != "mu2e_tranad" ]; then
    echo "WARNING: this repo must be cloned into a directory named mu2e_tranad (found: $(basename "$_PKG"))"
fi

export PATH="$_ANA/bin:$PATH"

if [ ! -d "$_EXTRAS/seaborn" ]; then
    python -m pip install --quiet --no-deps --target "$_EXTRAS" seaborn==0.13.2
fi

export PYTHONPATH="$_PARENT:$_EXTRAS${PYTHONPATH:+:$PYTHONPATH}"

# Mu2e/DQM python package, for running the `tranad` evaluator (dqm/evaluate/tranad.py)
# with dqm-driver from this env; a clone of github.com/Mu2e/DQM next to this repo
if [ -d "$_PARENT/DQM/python/dqm" ]; then
    export PYTHONPATH="$_PARENT/DQM/python:$PYTHONPATH"
fi
export MPLBACKEND=Agg
export PYTHONWARNINGS=ignore

python -c "import mu2e_tranad.training.tranad_model as t; assert t.TRANAD_ANL_AVAILABLE" 2>/dev/null \
    && echo "mu2e_tranad env ready (TranAD available)" \
    || echo "WARNING: TranAD not importable; run: python -c 'import mu2e_tranad.training.tranad_model'"

unset _PKG _PARENT _EXTRAS _ANA
