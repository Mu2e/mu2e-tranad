# Environment recipe for running mu2e_tranad (incl. TranAD) on a CPU Mu2e gpvm
# (e.g. mu2egpvm05) without the anldq conda env.
#
# Expected layout: this repo cloned as mu2e_tranad/ inside a CrossExperimentalAIDQM
# checkout, so that anldq is at ../src/anldq:
#   CrossExperimentalAIDQM/
#   ├── src/anldq/
#   └── mu2e_tranad/        <- this repo
#
# Usage (from anywhere):
#   source mu2e_tranad/setup_env.sh
#   python -m mu2e_tranad --dqm-source crvreco,kpp_crv,file,005-001 --output results/<dir>
#
# Base: Mu2e cvmfs "ana" env (Python 3.12, torch 2.5.1, pandas, sklearn, matplotlib).
# It is missing a few packages the pipeline/anldq import, so these are installed
# once into mu2e_tranad/.ana_extras and added to PYTHONPATH:
#   - seaborn      : used by analysis/visualization.py (correlation heatmap)
#   - tensorboard  : imported when anldq is imported
#   - spt3g (stub) : imported by anldq's SPT-3G dataset loader, which the DQM
#                    pipeline never uses; the real library is a compiled telescope
#                    package, so a no-op stand-in is enough here.
# Without these, anldq fails to import and TranAD is silently skipped
# (0 TranAD anomalies, no training curve plot).

_PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO="$(dirname "$_PKG")"
_EXTRAS="$_PKG/.ana_extras"
_ANA=/cvmfs/mu2e.opensciencegrid.org/env/ana/current

if [ "$(basename "$_PKG")" != "mu2e_tranad" ]; then
    echo "WARNING: this repo must be cloned into a directory named mu2e_tranad (found: $(basename "$_PKG"))"
fi
if [ ! -d "$_REPO/src/anldq" ]; then
    echo "WARNING: $_REPO/src/anldq not found; clone this repo inside a CrossExperimentalAIDQM checkout"
fi

export PATH="$_ANA/bin:$PATH"

if [ ! -d "$_EXTRAS/seaborn" ] || [ ! -d "$_EXTRAS/tensorboard" ]; then
    python -m pip install --quiet --no-deps --target "$_EXTRAS" \
        seaborn==0.13.2 tensorboard==2.18.0 tensorboard-data-server==0.7.2
fi

if [ ! -f "$_EXTRAS/spt3g/core.py" ]; then
    mkdir -p "$_EXTRAS/spt3g"
    echo "# stub: SPT-3G loader unused by the DQM pipeline" > "$_EXTRAS/spt3g/__init__.py"
    cat > "$_EXTRAS/spt3g/core.py" <<'EOF'
class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()

def __getattr__(name):
    return _Any
EOF
    cp "$_EXTRAS/spt3g/core.py" "$_EXTRAS/spt3g/calibration.py"
fi

export PYTHONPATH="$_REPO:$_EXTRAS${PYTHONPATH:+:$PYTHONPATH}"

# Mu2e/DQM python package (dqm.evaluate), for the online evaluator in
# mu2e_tranad/evaluator/; a clone of github.com/Mu2e/DQM at $_REPO/DQM
if [ -d "$_REPO/DQM/python/dqm" ]; then
    export PYTHONPATH="$_REPO/DQM/python:$PYTHONPATH"
fi
export MPLBACKEND=Agg
export PYTHONWARNINGS=ignore

python -c "import mu2e_tranad.training.tranad_model as t; assert t.TRANAD_ANL_AVAILABLE" 2>/dev/null \
    && echo "mu2e_tranad env ready (TranAD available)" \
    || echo "WARNING: TranAD not importable; run: python -c 'import mu2e_tranad.training.tranad_model'"

unset _PKG _REPO _EXTRAS _ANA
