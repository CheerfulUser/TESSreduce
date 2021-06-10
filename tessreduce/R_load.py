import numpy as np

R = {'g': {'coeff': [ 3.61562687, -0.0891928 ],
	  'std': 0.004146827352467696},
	 'r': {'coeff': [ 2.58602003, -0.03325315],
	  'std': 0.0010620316190595924},
	 'i': {'coeff': [ 1.90959054, -0.01284678],
	  'std': 0.0004962971568272631},
	 'z': {'coeff': [ 1.50168735, -0.0045642 ],
	  'std': 0.0014331914679903046},
	 'y': {'coeff': [ 1.25340149, -0.00247802],
	  'std': 0.0005840472105137083},
	 'kep': {'coeff': [ 2.68629375, -0.26884456],
	  'std': 0.0020136674269240393},
	 'tess': {'coeff': [ 1.902, -0.179],
	  'std': 0.0028265468011445124}}

def line(x, c1, c2): 
    return c1 + c2*x

def R_val(band,g=None,r=None,gr=None,ext=0):
	if (g is not None) & (r is not None):
		gr = g-r

	if (gr is None) | np.isnan(gr).all():
		Rb   = R[band]['coeff'][1]
	else:
		Rr0 = R[band]['coeff'][1]
		Rg0 = R[band]['coeff'][1]

		gr_int = gr - ext*(Rg0 - Rr0)

		vals = R[band]['coeff']
		Rb  = line(gr_int,vals[0],vals[1])
	Rb_e = R[band]['std']

	return Rb, Rb_e
