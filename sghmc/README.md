### Install

To install the package, run

```
$ git clone https://github.com/Senlody/SGHMC.git
$ cd SGHMC/sghmc
$ python setup.py install
```

You may need Administrator rights to install the package.

### examples

To run examples, `cd` to `SGHMC/sghmc/tests` folder, and run one of the following

```
$ python mixnormal.py
$ python simpleU.py
$ python bnnMPG.py
```

A successful run of example script ends without throwing any error.

WARNNING: `mixnormal.py` and `simpleU.py` contains `sghmc_chains` that only works on linux. Running these examples on windows will cause errors.

For the details of the examples, refer to project report in `SGHMC/report`.
