# microbiome-patterns

> **Vendored into the SSMC repo — this block is an SSMC note, not part of upstream.**
>
> Upstream: <https://github.com/Emergent-Behaviors-in-Biology/microbiome-patterns>
> at commit `7bbf8a32b30a9c293812bfabf1324f43b202014c` ("Update README.md"), which was
> the upstream tip when this copy was taken. The clone's own `.git` was deleted: this
> copy takes no updates and nothing is pushed back. To see what changed since, or to
> diff against pristine upstream, clone it fresh alongside and compare.
>
> **Modified from upstream:** `A Minimal Model for Microbial Biodiversity.ipynb` (+37/-14).
> **Added here:** `extract_micrm_params.py`, `loading_extracted_micrm_params.jl`,
> `Spatial linstab 1.ipynb`, `linstab_plots/`, `requirements-env1.txt`, `env1-freeze.txt`,
> and the `community-simulator/` copy described below.

## Data and environment kept out of git

Three directories are gitignored in SSMC and live on this disk only. Nothing here is
lost - each is either re-downloadable or regenerable:

| path | size | what it is |
|---|---|---|
| `data/` | 944 M | upstream's own data, tracked in the upstream repo |
| `env1/` | 469 M | Python 3.6.15 virtualenv for running the upstream notebook |
| `micrm_params_used/` | 217 M | 14 HDF5 files, output of `extract_micrm_params.py` |

**`data/`** - take it from upstream (the tip is the same `7bbf8a3` this copy came from;
if it has since moved, `git -C /tmp/mp checkout 7bbf8a3` before the move):

```bash
git clone https://github.com/Emergent-Behaviors-in-Biology/microbiome-patterns.git /tmp/mp
mv /tmp/mp/data ./data && rm -rf /tmp/mp
```

**`env1/`** - a 2019-era stack, deliberately period-correct for the paper rather than
current. `requirements-env1.txt` holds the pins *and* the two build-order caveats that
a plain `pip install -r` will not honour (cython before `scikit-bio`, and
`community_simulator` installed `--no-deps` from `./community-simulator`); follow its
comments rather than the file alone. Outline:

```bash
pyenv install -s 3.6.15
~/.pyenv/versions/3.6.15/bin/python -m venv env1
# then the pinned installs, per the comments in requirements-env1.txt
env1/bin/pip install h5py==2.10.0   # needed by extract_micrm_params.py
```

`env1-freeze.txt` is the exact realised state of the env, but it was written before
`h5py` was added, so it is one package short of what the extraction script needs.

**`micrm_params_used/`** - regenerate from `data/` once `env1` exists. No options, a few
minutes:

```bash
env1/bin/python extract_micrm_params.py
```
Scripts and data to accompany [Marsland, Cui and Mehta "A minimal model for microbial biodiversity can reproduce experimentally observed ecological patterns" _Scientific Reports_ **10**:3308 (2020)](https://www.nature.com/articles/s41598-020-60130-2)
