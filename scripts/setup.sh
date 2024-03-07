#!/bin/zsh

# Install Brew.
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Bazelisk.
brew install bazelisk

# Install pre-commit.
conda install -c conda-forge pre-commit

# Install the pre-commit hooks.
pre-commit install
