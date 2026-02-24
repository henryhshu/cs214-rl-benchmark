# cs214-rl-benchmark

## Setup
See `./pyproject.toml` for dependencies. While there are shared dependencies for torchforge and ray experiments, please be sure to manage dependencies in different environments. See below for example setup.

Setting up torchforge environment:
1. `python -m venv .venv-forge`
2. `source .venv-forge/bin/activate`
3. `pip install -e .[forge]`
4. Deactivate with `deactivate` as needed

Setting up Ray environment:
1. `python -m venv .venv-ray`
2. `source .venv-ray/bin/activate`
3. `pip install -e .[ray]`
4. Deactivate with `deactivate` as needed

Setting up OpenEnv:
1. Make sure you are in a relevant venv
2. `pip install -e external/OpenEnv/`

Running Blacjack Environment:
1. Open terminal instance and ensure cwd is this repo
2. Switch to a relevant venv
3. `export PYTHONPATH="$(pwd)/external/OpenEnv:$PYTHONPATH"`
4. `OPENSPIEL_GAME=blackjack python -m envs.openspiel_env.server.app --port 8004`
5. Kill process as needed

> Note: OpenEnv envs module not added via pip install so workarounds are used.