import numpy as np

# Dictionary of Pan Stars 1 R values
R_ps1 = {'g': {'coeff': [3.61562687, -0.0891928],
               'std': 0.004146827352467696},
         'r': {'coeff': [2.58602003, -0.03325315],
               'std': 0.0010620316190595924},
         'i': {'coeff': [1.90959054, -0.01284678],
               'std': 0.0004962971568272631},
         'z': {'coeff': [1.50168735, -0.0045642],
               'std': 0.0014331914679903046},
         'y': {'coeff': [1.25340149, -0.00247802],
               'std': 0.0005840472105137083},
         'kep': {'coeff': [2.68629375, -0.26884456],
                 'std': 0.0020136674269240393},
         'tess': {'coeff': [1.902, -0.179],
                  'std': 0.0028265468011445124}
         }


# Dictionary of Skymapper R values
R_sm = {'u': {'coeff': [4.902198196976004, -0.04396865635703249],
              'std': 0.005212225623300655},
        'v': {'coeff': [4.553419148586131, -0.03096904487746069],
              'std': 0.015102362684553196},
        'g': {'coeff': [3.434788880230338, -0.17247389098523408],
              'std': 0.0019526614969365428},
        'r': {'coeff': [2.6377280770536853, -0.07696556583546744],
              'std': 0.0016588895668870856},
        'i': {'coeff': [1.8190330572341713, -0.01977796422745485],
              'std': 0.0008544792739952313},
        'z': {'coeff': [1.3827366254049507, -0.017314195591388342],
              'std': 0.003027218953684425},
        'tess': {'coeff': [1.902, -0.179],
                 'std': 0.0028265468011445124}
        }


def line(x, c1, c2):
    """
    Makes a line based on a y-intercept and a slope:

    y = c1 + c2*x
    -------------

    Paramters:
    ---------
    x: array_like  
            An array of x values to be turned into points on the line.
    c1: float
            The y-intercept of the line.  
    c2: float
            The slope of the line.

    Returns:
    -------
    this_line: array_like
            The line from the specified parameters, the same shape as x. 
    """
    this_line = c1+c2*x
    return this_line


def R_val(band, g=None, r=None, gr=None, ext=0, system='ps1'):
    """
    Calcuates the R values based on the system that is used. 

    Parameters:
    ----------
    band: str
            The band that the extinction should be calculated for.
    g: float, optional
            The g mag of the object. Default is None.
    r:
            The r mag of the object. Default is None.
    gr: float, optional
            The g-r colour of the object, is superseeded by g-r if g and r are given. Default is None.
    ext: int
            Default is 0.
    system: str
            Values checked for are 'ps1' and 'skymapper'. Other strings should not be used. Default is 'ps1'.

    Returns:
    -------
    Rb:
            The R value through the band, from known dictionaries.
    Rb_e:
            The standard deviation of Rb, from known dictionaries.. 

    """
    if system.lower() == 'ps1':
        R = R_ps1
    elif system.lower() == 'skymapper':
        R = R_sm
    if (g is not None) & (r is not None):
        gr = g-r

    if (gr is None) | np.isnan(gr).all():
        Rb = R[band]['coeff'][1]
    else:
        Rr0 = R[band]['coeff'][1]
        Rg0 = R[band]['coeff'][1]

        gr_int = gr - ext*(Rg0 - Rr0)

        vals = R[band]['coeff']
        Rb = line(gr_int, vals[0], vals[1])
    Rb_e = R[band]['std']

    return Rb, Rb_e
