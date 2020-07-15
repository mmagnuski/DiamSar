# DiamSar
code accompanying "Three times NO: no relationship between frontal alpha asymmetry and depressive disorders in a multiverse analysis of three studies" paper (you can find the preprint at bioRxiv: https://www.biorxiv.org/content/10.1101/2020.06.30.180760v1).

## Installation
Currently, the code is used without standard installation process - you just need to download the `DiamSar` repository, place the folder somewhere on your computer and add that location to the python path.
You can either add this location to the `PYTHONPATH` envirionment variable or add the path through python every time you use `DiamSar`:
```python
import sys
sys.path.append(r'C:\src\DiamSar')

import DiamSar as ds
```

## Folder structure
`DiamSar` expects a specific directory stucture that we used throught the analysis. The data should be available on Dryad soon - then a function will be added to DiamSar to create the expected directory structure from the provided data.
