import numpy as np
import blosc

import time
import sys

# ~ returns Python bytes object
def compress(grad):
    assert isinstance(grad, np.ndarray)
    # ~ pack (compress) a NumPy array to a Python bytes object.
    compressed_grad = blosc.pack_array(grad, cname='snappy')
    return compressed_grad

# ~ returns np.ndarray
def decompress(msg):
    if sys.version_info[0] < 3:
        # Python 2.x implementation
        assert isinstance(msg, str)
    else:
        # Python 3.x implementation
        assert isinstance(msg, bytes)
        
    # ~ unpack (decompress) a packed NumPy array to np.ndarray
    grad = blosc.unpack_array(msg)
    return grad