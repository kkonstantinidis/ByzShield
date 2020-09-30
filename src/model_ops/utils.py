# ~ these are the adversary attack models for the returned gradient
import numpy as np

ADVERSARY_=-100
CONST_ = -100

# ~ the type of float to be used for the gradients
float_type = np.float32
# float_type = np.float64

# ~ "*" is element-wise multiplication in numpy
def err_simulation(grad, mode, cyclic=False):
    if mode == "rev_grad":
        if cyclic:
            adv = ADVERSARY_*grad
            assert adv.shape == grad.shape
            return np.add(adv, grad)
        else:
            return ADVERSARY_*grad
    elif mode == "constant":
        if cyclic:
            adv = np.ones(grad.shape, dtype=float_type)*CONST_
            assert adv.shape == grad.shape
            return np.add(adv, grad)
        else:
            return np.ones(grad.shape, dtype=float_type)*CONST_
    #elif mode == "":
    elif mode == "random":
        # TODO(hwang): figure out if this if necessary
        return grad
    elif mode == "foe":
        return grad