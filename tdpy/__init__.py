import sys
import numpy as np

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from .util import *

