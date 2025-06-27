## madevent7

### Installation

#### Binary wheels

Binary wheels based on the last commit that passes all unit tests are built automatically
for Linux and MacOS X (with Apple silicon), for Python 3.11 and 3.12. Use one of the
following commands based on your OS and Python version:

```sh
# Linux, Python 3.11
pip install https://github.com/madgraph-ml/madevent7/releases/download/latest/madevent7-0.1.0-cp311-cp311-linux_x86_64.whl

# Linux, Python 3.12
pip install https://github.com/madgraph-ml/madevent7/releases/download/latest/madevent7-0.1.0-cp312-cp312-linux_x86_64.whl

# MacOS X, Python 3.11
pip install https://github.com/madgraph-ml/madevent7/releases/download/latest/madevent7-0.1.0-cp311-cp311-macosx_14_0_arm64.whl

# MacOS X, Python 3.12
pip install https://github.com/madgraph-ml/madevent7/releases/download/latest/madevent7-0.1.0-cp312-cp312-macosx_14_0_arm64.whl
```

#### Development version

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
pip install --no-build-isolation -Cbuild-dir=build -Ccmake.build-type=RelWithDebInfo .
```

This will create a directory `build` where you can run make directly to make development
easier. To update the python module itself, make sure to also run the `pip install` command
above again. This will not happen automatically, even if you make the installation editable!
Build type `RelWithDebInfo` generates optimized code but includes debug symbols, so you
can use `lldb` or `gdb` to debug the code.

### Tests

To run the tests, you need to have the `pytest`, `numpy` and `torch` packages installed.
One test optionally requires the `lhapdf` package (can be installed via conda or built from
source) and the `NNPDF40_nlo_as_01180` PDF set.

To run the tests, go to the root directory of the repository and run
```sh
pytest tests
```
