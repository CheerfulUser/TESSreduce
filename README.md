# TESS TPF reduction

With these functions you can download an FFI cuout from TESScut and reduce the TPF data. The background subtraction accounts for the smooth background and
detector straps. If "shift" is true, then all images will be shifted to match a reference image. 

TESSreduce can be installed through pip:
pip install +git@github.com:CheerfulUser/TESSreduce.git

To run:
import tpf_reduction
download tpf with Get_TESS
run Quick_reduce

For more information contact me at: rridden @ stsci.edu
