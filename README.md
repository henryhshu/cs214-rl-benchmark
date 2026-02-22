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