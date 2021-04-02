# TESS TPF reduction

With these functions you can download an FFI cuout from TESScut and reduce the TPF data. The background subtraction accounts for the smooth background and
detector straps. If "shift" is true, then all images will be shifted to match a reference image. 

TESSreduce can be installed through pip:

`pip install git+https://github.com/CheerfulUser/TESSreduce.git`

Example reduction for SN 2020fqv:
```
import tpf_reduction as tr
ra = 189.1385817
dec = 11.2316535
tess = tr.tessreduce(ra=ra,dec=dec)
tess.reduce()
# If you want to remove residual background trends as best as possible
detrend = tess.detrend_transient()
```
![plot](./figs/detrend_comparison.png)

For more information contact me at: rridden @ stsci.edu
