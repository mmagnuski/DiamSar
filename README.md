# DiamSar
code accompanying "Three times no: no relationship between frontal alpha asymmetry and depressive disorders in a multiverse analysis of three studies" paper (you can find the preprint on bioRxiv: https://www.biorxiv.org/content/10.1101/2020.06.30.180760v1).

## Required packages
You need a standard sientific python istallation (Anaconda is recommended) with `numpy`, `scipy`, `pandas`, `matplotlib` etc. Additional non-standard libraries are:
* [`mne-python`](https://github.com/mne-tools/mne-python), preferably version `0.19`, which was used in the "Three times no" paper.
* [`borsar`](https://github.com/mmagnuski/borsar)
* [`sarna`](https://github.com/mmagnuski/sarna)

To plot the figure 1 supplement 1 or summarize channel-pair analyses in a table with effect sizes and their bootstrap confidence intervals you will also need the following packages:
* [`pingouin`](https://github.com/raphaelvallat/pingouin)
* [`scikits.bootstrap`](https://github.com/cgevans/scikits-bootstrap)
* [`DABEST-python`](https://github.com/ACCLAB/DABEST-python)

## Installation
Currently, the code is used without standard installation process - you just need to download the `DiamSar` repository, place the folder somewhere on your computer and add that location to the python path.
You can either add this location to the `PYTHONPATH` envirionment variable or add the path through python every time you use `DiamSar`:
```python
import sys
sys.path.append(r'C:\src\DiamSar')

import DiamSar as ds
```

## Folder structure
`DiamSar` expects a specific directory stucture that we used throught the analysis. The data should be available on Dryad soon - they contain one zip package for each study. To use `DiamSar` code you need at least the DiamSar study unzipped somewhere on your computer. However to reproduce the analyses from the paper it is best to download and unzip all  three studies.  In the end you should have a folder with three subfolders: `DiamSar`, `Wronski` and `Nowowiejska` (these are studies III, II and I, respectively).

## Usage example
(more examples will come soon in the form of notebooks)  
Once you have the folder structure set up, assuming you have the studies located in `C:\data\threetimesno` directory, to import and activate DiamSar you need to execute:
```python
import DiamSar as ds
paths = ds.pth.set_paths(base_dir=r'C:\data\threetimesno')
```

Now the `paths` variable contains a `borsar.project.Paths` object that allows to easily get paths and data for any of the three studies. For example you can read BDI scores for the participants for given study in the following way:
```python
bdi = paths.get_data('bdi', study='B')
```
or read power spectra for given study-space combination:
```python
psd, freq, ch_names, subj_id = paths.get_data('psd', study='C', space='avg')
```

Now the easiest analysis you can perform is with the deault settings (to learn more see the documentation of `DiamSar.analysis.run_analysis`):
```python
# run the default analysis
clst = ds.analysis.run_analysis()

# plt the topography of the statistics (t test in this case)
topo = clst.plot(vmin=-2, vmax=2)
# add colorbar
plt.colorbar(topo.img)
```
