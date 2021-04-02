![plot](./figs/header.png)

With this package that builds on lightkurve, you can reduce TESS data while preserving transient signals. You can supply a TPF or give coordinates to construct a TPF with TESScut.The background subtraction accounts for the smooth background and
detector straps. Alongisde background subtraction TESSreduce also aligns images, performs difference imaging, and can even detect transient events! 

An additional component that is in development is calibration of TESS photometry, and reliably link muti-sector light curves.

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


There are a lot of other functions burried in there which currently aren't well documented, so for more information contact me at: rridden@stsci.edu
