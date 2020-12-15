import numpy as np
import numpy.linalg as npla

def cross_op(x):
  """
  Vectorized
  compute x^cross, the skew symmetric matrix
    x: ...x3x1 vector
  """
  x_cross = np.zeros(x.shape[:-2] + (3, 3))
  x_cross[..., 0, 1] = -x[..., 2, 0]
  x_cross[..., 0, 2] = x[..., 1, 0]
  x_cross[..., 1, 0] = x[..., 2, 0]
  x_cross[..., 1, 2] = -x[..., 0, 0]
  x_cross[..., 2, 0] = -x[..., 1,0 ]
  x_cross[..., 2, 1] = x[..., 0, 0]
  return x_cross

def wedge_op(x):
  return cross_op(x)

def vee_op(x):
  x_vee = np.zeros(x.shape[:-2] + (3, 1))
  x_vee[..., 0, 0] = x[..., 2, 1]
  x_vee[..., 1, 0] = x[..., 0, 2]
  x_vee[..., 2, 0] = x[..., 1, 0]
  return x_vee

def psi_to_C(psi):
  """
  Vectorized
  axis angle to rotation matrix
    psi: ...x3x1 vector
  """
  ag = npla.norm(psi, axis = (-2, -1), keepdims=True)
  ax = psi / ag
  eye = np.zeros(ag.shape[:-2] + (3,3))
  eye[..., :, :] = np.eye(3)
  C = np.cos(ag) * eye + (1 - np.cos(ag)) * (ax @ ax.swapaxes(-2, -1)) - np.sin(ag) * cross_op(ax)
  return C