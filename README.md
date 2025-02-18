## madevent7

### Installation

First install `scikit_build_core` with

```sh
pip install scikit_build_core
```

The pre-installed version of `cmake` under MacOS is outdated, so you might need to install a
newer version, for example with

```sh
brew install cmake
```

Then check out the `madevent7` repository and build and install it with

```sh
git clone git@github.com:madgraph-ml/madevent7.git
cd madevent7
pip install --no-build-isolation -Cbuild-dir=build .
```

This will create a directory `build` where you can run make directly to make development easier.
To update the python module itself, make sure to also run the `pip install` command above again.
This will not happen automatically, even if you make the installation editable!
