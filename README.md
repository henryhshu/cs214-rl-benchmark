# cs214-rl-benchmark

## Setup
See `./pyproject.toml` for dependencies. While there are shared dependencies for torchforge and ray experiments, please be sure to manage dependencies in different environments. See below for example setup.

Setting up torchforge environment:
1. `python -m venv .venv-forge`
2. `source .venv-forge/bin/activate`
3. `pip install -e .[forge]`
4. Deactivate with `deactivate`

Setting up Ray environment:
1. `python -m venv .venv-ray`
2. `source .venv-ray/bin/activate`
3. `pip install -e .[ray]`
4. Deactivate with `deactivate`

Setting up OpenEnv (for Blackjack):
1. `git clone https://github.com/meta-pytorch/OpenEnv.git ../OpenEnv`
2. `cd ../OpenEnv`
3. `pip install -e .`
4. Switch to new terminal instance
5. `export OPENENV_PATH="{your path}/OpenEnv"`
6. `export PYTHONPATH="${OPENENV_PATH}:${PYTHONPATH}"`
7. `OPENSPIEL_GAME=blackjack python -m envs.openspiel_env.server.app --port 8004`

> Note: OpenEnv doesn't put envs submodules under src directory, this might change looking at issues on GitHub. This would effect the need to update system paths.