import numpy as np
import numpy.linalg as npla
import scipy.linalg as cpla

import so3

def Cr2T(C_a_b, r_b_a_b):
  """
  Vectorized but not broadcastable
  Rotation matrix and translation vector to pose matrix
    C_{ab}: ...x3x3 matrix
    r_{b}^{ab}: ...x3x1 matrix
  """
  assert C_a_b.shape[:-2] == r_b_a_b.shape[:-2]

  r_a_b_a = -C_a_b @ r_b_a_b
  T_a_b = np.zeros(C_a_b.shape[:-2] + (4, 4))
  T_a_b[..., :3, :3] = C_a_b
  T_a_b[..., :3, 3:4] = r_a_b_a
  T_a_b[..., 3, 3] = 1
  return T_a_b

def T2Cr(T_a_b):
  """
  Vectorized but not broadcastable
  pose matrix to rotation matrix and translation vector 
    T_{ab}: ...x4x4 matrix
  """
  r_a_b_a = T_a_b[..., :3, 3:4]
  C_a_b = T_a_b[..., :3, :3]
  r_b_a_b = -C_a_b.swapaxes(-2, -1) @ r_a_b_a
  return C_a_b, r_b_a_b

def expm(x):
  if len(x.shape) == 2:
    return cpla.expm(x)
  else:
    shape = x.shape
    x = x.reshape(-1, *x.shape[-2:])
    expx = np.zeros_like(x)
    for i in range(x.shape[0]):
      expx[i] = cpla.expm(x[i])
    expx.reshape(shape)
    return expx

def logm(x):
  if len(x.shape) == 2:
    return cpla.logm(x)
  else:
    shape = x.shape
    x = x.reshape(-1, *x.shape[-2:])
    logx = np.zeros_like(x)
    for i in range(x.shape[0]):
      logx[i] = cpla.logm(x[i])
    logx.reshape(shape)
    return logx


def wedge_op(x):
  """
  Vectorized
    x: ...x6x1 vector
  """
  x_wedge = np.zeros(x.shape[:-2] + (4, 4))
  x_wedge[..., :3, :3] = so3.wedge_op(x[..., 3:, :])
  x_wedge[..., :3, 3:4] = x[..., :3, :]
  return x_wedge


def vee_op(x):
  """
  Vectorized
    x: ...x4x4 matrix
  """
  x_vee = np.zeros(x.shape[:-2] + (6, 1))
  x_vee[..., 3:, :] = so3.vee_op(x[..., :3, :3])
  x_vee[..., :3, :] = x[..., :3, 3:4]
  return x_vee

def curly_wedge_op(x):
  """
  Vectorized
    x: ...x6x1 vector
  """
  x_curly_wedge = np.zeros(x.shape[:-2] + (6, 6))
  x_curly_wedge[..., :3, :3] = so3.wedge_op(x[..., 3:, :])
  x_curly_wedge[..., 3:, 3:] = so3.wedge_op(x[..., 3:, :])
  x_curly_wedge[..., :3, 3:] = so3.wedge_op(x[..., :3, :])
  return x_curly_wedge

def odot_op(p):
  """
  Vectorized
    p: ...x4x1 vector
  """
  eye = np.zeros(p.shape[:-2] + (3,3))
  eye[..., :, :] = np.eye(3)
  eps = p[..., :-1, :]
  eta = p[..., 3, 0]
  eta = eta.reshape(*eta.shape, *([1] * len(eye.shape[-2:])))
  p_odot = np.zeros(p.shape[:-2] + (4, 6))
  p_odot[..., :3, :3] = eta * eye
  p_odot[..., :3, 3:] = -so3.wedge_op(eps)
  return p_odot


def Ad(T):
  Ad_T = np.zeros(T.shape[:-2] + (6, 6))
  C = T[..., :3, :3]
  r = T[..., :3, 3:4]
  Ad_T[..., :3, :3] = C
  Ad_T[..., 3:, 3:] = C
  Ad_T[..., :3, 3:] = so3.cross_op(r) @ C
  return Ad_T

if __name__ == "__main__":
  # test Cr2T
  C = np.zeros((5, 4, 3, 3))
  r = np.zeros((5, 4, 3, 1))
  T = Cr2T(C, r)
  C = np.zeros((3, 3))
  r = np.zeros((3, 1))
  T = Cr2T(C, r)  
  print(T)
