# cs214-rl-benchmark

## Setup

### Prereqs
- Conda
- Patience
- Lots of storage (torchforge estimates ~50GB but tied to models)

### Managing Environment and Dependencies
1. `conda env create -f environment.yaml`
2. `cd external/torchforge`
3. `./scripts/install.sh`
    a. See [torchforge repo](https://github.com/meta-pytorch/torchforge/tree/main) for source
4. `conda deactivate && conda activate rl-benchmark`


### Running Blacjack Environment:
1. Open terminal instance and ensure cwd is this repo
2. Switch to a relevant venv
3. `export PYTHONPATH="$(pwd)/external/OpenEnv:$PYTHONPATH"`
    - export PYTHONPATH="/Users/henry/cs214-rl-benchmark/external/OpenEnv:/Users/henry/cs214-rl-benchmark/src"
4. `OPENSPIEL_GAME=blackjack python -m envs.openspiel_env.server.app --port 8004`
5. Kill process as needed

> Note: OpenEnv envs module not added via pip install so workarounds are used.