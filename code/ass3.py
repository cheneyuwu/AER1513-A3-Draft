import time
import numpy as np
import numpy.linalg as npla
import scipy.linalg as cpla
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt

from pylgmath import so3op, se3op, Transformation
from pysteam import state, evaluator, problem, solver

## Configure matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize


class Estimator:

  def __init__(self, dataset):
    # load data
    data = loadmat(dataset)

    # total time steps
    self.K = data["t"].shape[-1]

    # stereo camera
    self.f_u = data["fu"][0, 0]
    self.f_v = data["fv"][0, 0]
    self.c_u = data["cu"][0, 0]
    self.c_v = data["cv"][0, 0]
    self.b = data["b"][0, 0]

    # stereo camera and imu
    C_cv, rho_cv_inv = data["C_c_v"], data["rho_v_c_v"]
    self.T_cv = se3op.Cr2T(C_cv, -C_cv @ rho_cv_inv)

    # ground truth values
    r_vi_ini = data["r_i_vk_i"].T[..., None]
    C_vi = so3op.vec2rot(data["theta_vk_i"].T[..., None]).swapaxes(-1, -2)
    self.T_vi = se3op.Cr2T(C_vi, -C_vi @ r_vi_ini)  # this is the ground truth

    # inputs
    w_vi_inv, v_vi_inv = data["w_vk_vk_i"].T, data["v_vk_vk_i"].T
    self.varpi_iv_inv = np.concatenate([-v_vi_inv, -w_vi_inv], axis=-1)[..., None]
    self.t = data["t"].squeeze()  # time steps (1900,)
    ts = np.roll(self.t, 1)
    ts[0] = 0
    self.dt = self.t - ts

    # measurements
    rho_pi_ini = data["rho_i_pj_i"].T[..., None]  # feature positions (20 x 3 x 1)
    rho_pi_ini = np.repeat(rho_pi_ini[None, ...], self.K, axis=0)  # feature positions (1900, 20 x 3 x 1)
    padding = np.ones(rho_pi_ini.shape[:-2] + (1,) + rho_pi_ini.shape[-1:])
    self.rho_pi_ini = np.concatenate((rho_pi_ini, padding), axis=-2)  # feature positions (1900, 20 x 4 x 1)
    self.y_k_j = data["y_k_j"].transpose((1, 2, 0))[..., None]  # measurements (1900, 20, 4, 1)
    self.y_filter = np.where(self.y_k_j == -1, 0, 1)  # [..., 0, 0] # filter (1900, 20, 4, 1)

    # covariances
    w_var, v_var, y_var = data["w_var"], data["v_var"], data["y_var"]
    w_var_inv = np.reciprocal(w_var.squeeze())
    v_var_inv = np.reciprocal(v_var.squeeze())
    y_var_inv = np.reciprocal(y_var.squeeze())
    self.Q_inv = np.zeros((self.K, 6, 6))
    self.Q_inv[..., :, :] = cpla.block_diag(np.diag(v_var_inv), np.diag(w_var_inv))
    self.R_inv = np.zeros((*(self.y_k_j.shape[:2]), 4, 4))
    self.R_inv[..., :, :] = np.diag(y_var_inv)

    # helper matrices
    self.D = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # estimated values of variables
    self.hat_T_vi = np.zeros_like(self.T_vi)
    self.hat_T_vi[...] = self.T_vi[...]  # estimate of poses initialized to be ground truth
    self.hat_P = np.zeros((self.K, 6, 6))
    self.hat_P[..., :, :] = np.eye(6) * 1e-4
    self.hat_stds = np.ones((self.K, 6)) * np.sqrt(1e-4)
    # copy for initial prior values (opdated by optimize function)
    self.init_T_vi = np.zeros_like(self.hat_T_vi)
    self.init_T_vi[...] = self.hat_T_vi[...]
    self.init_P = np.zeros_like(self.hat_P)
    self.init_P[...] = self.hat_P[...]

    self.k1 = 0
    self.k2 = self.K

    # Timing
    self.optimization_time = 0

  def set_interval(self, k1=None, k2=None):
    self.k1 = k1 if k1 != None else self.k1
    self.k2 = k2 if k2 != None else self.k2

  def initialize(self, k1=None, k2=None):
    """
    Initialize a portion of the states between k1 and k2 using dead reckoning
    and starting with the current estimate of k1

    Note: we initialize our estimate with ground truth (hack) in constructor
    call
    """
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    # self.hat_T_vi[k1] = self.T_vi[k1]  # force to ground truth
    # self.hat_P[k1] = 1e-3 * np.eye(6)  # force to ground truth
    for k in range(k1 + 1, k2):
      # TODO: need to check initialization, input time step is strange (but same as in assignment)
      # mean
      self.hat_T_vi[k] = self.f(self.hat_T_vi[k - 1], self.varpi_iv_inv[k - 1], self.dt[k])
      # covariance
      F = self.df(self.hat_T_vi[k - 1], self.varpi_iv_inv[k - 1], self.dt[k])
      Q_inv = self.Q_inv[k]
      Q_inv = Q_inv / (self.dt[k, None, None]**2)
      self.hat_P[k] = F @ self.hat_P[k - 1] @ F.T + npla.inv(Q_inv)

    # this is only for the initial error term
    self.init_T_vi[...] = self.hat_T_vi[...]
    self.init_P[...] = self.hat_P[...]

  def optimize(self, k1=None, k2=None):
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    start_time = time.time()

    curr_iter, eps = 0, np.inf
    while curr_iter < 20 and eps > 1e-5:
      eps = self.update(k1, k2)
      curr_iter += 1
      print('GN step: {}   eps: {}'.format(curr_iter, eps))

    self.optimization_time += time.time() - start_time

    # this is only for the initial error term
    self.init_T_vi[...] = self.hat_T_vi[...]
    self.init_P[...] = self.hat_P[...]

  def update(self, k1=None, k2=None):
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    # First input factor
    # error
    T = self.hat_T_vi[k1]
    T_prior = self.init_T_vi[k1]
    e_v0 = self.e_v0(T_prior, T)
    # Jacobian
    H_v0 = np.zeros((6, (k2 - k1) * 6))
    H_v0[:6, :6] = np.eye(6)
    # covariance
    P0_inv = npla.inv(self.init_P[k1])

    # Subsequent input errors
    T = self.hat_T_vi[k1:k2 - 1]
    T2 = self.hat_T_vi[k1 + 1:k2]
    v = self.varpi_iv_inv[k1:k2 - 1]
    dt = self.dt[k1 + 1:k2]
    # error
    e_v = self.e_v(T2, T, v, dt).reshape(-1, 1)
    # Jacobian
    F = self.F(T2, T, v, dt)
    H_v = np.zeros(((k2 - k1 - 1) * 6, (k2 - k1) * 6))
    for i in range(F.shape[0]):
      H_v[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)] = -F[i]
      H_v[6 * i:6 * (i + 1), 6 * (i + 1):6 * (i + 2)] = np.eye(6)
    # covariance
    Q_inv = self.Q_inv[k1 + 1:k2]
    Q_inv = Q_inv / (dt[..., None, None]**2)
    Q_inv = cpla.block_diag(*Q_inv)

    # Measurement errors
    p = self.rho_pi_ini[k1:k2]
    y = self.y_k_j[k1:k2]
    T = np.repeat(self.hat_T_vi[k1:k2][:, None, ...], self.y_k_j.shape[1], axis=1)
    # error
    e_y = self.e_y(y, p, T).reshape(-1, 1)
    # Jacobian
    G = self.G(y, p, T)
    H_y = np.zeros((np.prod(G.shape[0:3]), (k2 - k1) * 6))
    nrow = np.prod(G.shape[1:3])
    for i in range(G.shape[0]):
      H_y[nrow * i:nrow * (i + 1), 6 * i:6 * (i + 1)] = G[i].reshape(-1, 6)
    # covariance
    R_inv = self.R_inv[k1:k2]
    R_inv = R_inv.reshape(-1, 4, 4)
    R_inv = cpla.block_diag(*R_inv)
    # filter out invalid measurements
    mask = self.y_filter[k1:k2].reshape(-1)
    mask = np.argwhere(mask).squeeze()
    e_y = e_y[mask]
    H_y = H_y[mask]
    R_inv = R_inv[mask][:, mask]

    # Stack all the factors
    e = np.concatenate((e_v0, e_v, e_y), axis=0)
    H = np.concatenate((H_v0, H_v, H_y), axis=0)
    W_inv = cpla.block_diag(P0_inv, Q_inv, R_inv)
    # e = np.concatenate((e_v, e_y), axis = 0)
    # H = np.concatenate((H_v, H_y), axis=0)
    # W_inv = cpla.block_diag(Q_inv, R_inv)

    # Solve the linear system
    LHS = H.T @ W_inv @ H
    RHS = H.T @ W_inv @ e
    update = cpla.cho_solve(cpla.cho_factor(LHS), RHS)
    eps = npla.norm(update)

    # Update each pose
    # mean
    T = self.hat_T_vi[k1:k2]
    update = update.reshape(T.shape[0], 6, 1)
    self.hat_T_vi[k1:k2] = se3op.vec2tran(update) @ T
    # Covariance
    full_hat_P = npla.inv(LHS)
    self.hat_P[k1:k2] = np.array(
        [full_hat_P[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] for i in range(int(full_hat_P.shape[0] / 6))])
    # for plotting
    self.hat_stds[k1:k2] = (np.sqrt(np.diag(full_hat_P))).reshape((-1, 6))

    return eps

  def plot_trajectory(self, k1=None, k2=None):
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    C_vi, r_iv_inv = se3op.T2Cr(self.T_vi)
    r_vi_ini = -C_vi.swapaxes(-2, -1) @ r_iv_inv
    hat_C_vi, hat_r_iv_inv = se3op.T2Cr(self.hat_T_vi)
    hat_r_vi_ini = -hat_C_vi.swapaxes(-2, -1) @ hat_r_iv_inv

    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hat_r_vi_ini[:, 0], hat_r_vi_ini[:, 1], hat_r_vi_ini[:, 2], s=0.1, c='blue', label='estimate')
    ax.scatter(r_vi_ini[:, 0], r_vi_ini[:, 1], r_vi_ini[:, 2], s=0.1, c='blue', label='ground truth')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_xlim3d(0, 5)
    ax.set_ylim3d(0, 5)
    ax.set_zlim3d(0, 3)
    ax.legend()
    # plt.show()

  def plot_num_visible_landmarks(self):
    num_meas = np.sum(self.y_filter[..., :, 0, 0], axis=-1)
    green = np.argwhere(num_meas >= 3)
    red = np.argwhere(num_meas < 3)

    fig = plt.figure()
    fig.set_size_inches(8, 3)
    fig.subplots_adjust(left=0.1, bottom=0.2)
    ax = fig.add_subplot(111)
    ax.scatter(self.t[green], num_meas[green], s=1, c='green')
    ax.scatter(self.t[red], num_meas[red], s=1, c='red')
    ax.set_xlabel(r't [$s$]')
    ax.set_ylabel(r'Number of Visible Landmarks')
    fig.savefig('num_visible.png')
    # plt.show()

  def plot_error(self, filename, k1=None, k2=None):
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    C_vi, r_iv_inv = se3op.T2Cr(self.T_vi)
    r_vi_ini = -C_vi.swapaxes(-2, -1) @ r_iv_inv
    hat_C_vi, hat_r_iv_inv = se3op.T2Cr(self.hat_T_vi)
    hat_r_vi_ini = -hat_C_vi.swapaxes(-2, -1) @ hat_r_iv_inv

    eye = np.zeros_like(C_vi)
    eye[..., :, :] = np.eye(3)
    rot_err = so3op.hatinv(eye - hat_C_vi @ npla.inv(C_vi))
    trans_err = hat_r_vi_ini - r_vi_ini

    t = self.t[k1:k2]
    stds = self.hat_stds[k1:k2, :]

    # plot landmarks for reference
    num_meas = np.sum(self.y_filter[k1:k2, :, 0, 0], axis=-1)
    green = np.argwhere(num_meas >= 3)
    red = np.argwhere(num_meas < 3)

    plot_number = 711
    fig = plt.figure()
    fig.set_size_inches(8, 12)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.1, top=0.95, wspace=0.7, hspace=0.6)

    plt.subplot(plot_number)
    plt.scatter(t[green], num_meas[green], s=1, c='green')
    plt.scatter(t[red], num_meas[red], s=1, c='red')
    plt.xlabel(r't [$s$]')
    plt.ylabel(r'Num. of Visible L.')

    labels = ['x', 'y', 'z']
    for i in range(3):
      plt.subplot(plot_number + 1 + i)
      plt.plot(t, trans_err[k1:k2, i].flatten(), '-', linewidth=1.0)
      plt.plot(t, 3 * stds[:, i], 'r--', linewidth=1.0)
      plt.plot(t, -3 * stds[:, i], 'g--', linewidth=1.0)
      plt.fill_between(t, -3 * stds[:, i], 3 * stds[:, i], alpha=0.2)
      plt.xlabel(r"$t$ [$s$]")
      plt.ylabel(r"$\hat{r}_x - r_x$ [$m$]".replace("x", labels[i]))
    for i in range(3):
      plt.subplot(plot_number + 4 + i)
      plt.plot(t, rot_err[k1:k2, i].flatten(), '-', linewidth=1.0)
      plt.plot(t, 3 * stds[:, 3 + i], 'r--', linewidth=1.0)
      plt.plot(t, -3 * stds[:, 3 + i], 'g--', linewidth=1.0)
      plt.fill_between(t, -3 * stds[:, 3 + i], 3 * stds[:, 3 + i], alpha=0.2)
      plt.xlabel(r"$t$ [$s$]")
      plt.ylabel(r"$\hat{\theta}_x - \theta_x$ [$rad$]".replace("x", labels[i]))

    fig.savefig('{}.png'.format(filename))
    # plt.show()
    # plt.close()

  def f(self, T, v, dt):
    """
    Vectorized
    motion model
    """
    dt = dt.reshape(-1, *([1] * len(v.shape[1:])))
    return se3op.vec2tran(dt * v) @ T

  def df(self, T, v, dt):
    """
    Vectorized
    linearized motion model
    """
    dt = dt.reshape(-1, *([1] * len(v.shape[1:])))
    return se3op.vec2jacinv(dt * v)

  def e_v0(self, T_prior, T):
    """
    Vectorized
    initial error
    """
    return se3op.tran2vec(T_prior @ npla.inv(T))

  def e_v(self, T2, T, v, dt):
    """
    Vectorized
    the motion error given states at two time steps and input
    """
    return se3op.tran2vec(self.f(T, v, dt) @ npla.inv(T2))

  def F(self, T2, T, v, dt):
    """
    Vectorized
    F matrix between two poses
    """
    return se3op.tranAd(T2 @ npla.inv(T))

  def e_y(self, y, p, T):
    """
    Vectorized
    e matrix measurement
    """
    z = self.D @ self.T_cv @ T @ p
    g = np.zeros(z.shape[:-2] + (4, 1))
    g[..., 0, 0] = self.f_u * z[..., 0, 0] / z[..., 2, 0] + self.c_u
    g[..., 1, 0] = self.f_v * z[..., 1, 0] / z[..., 2, 0] + self.c_v
    g[..., 2, 0] = self.f_u * (z[..., 0, 0] - self.b) / z[..., 2, 0] + self.c_u
    g[..., 3, 0] = self.f_v * z[..., 1, 0] / z[..., 2, 0] + self.c_v
    return y - g

  def G(self, y, p, T):
    """
    Vectorized
    G matrix measurement
    """
    z = self.D @ self.T_cv @ T @ p
    dgdz = np.zeros(z.shape[:-2] + (4, 3))
    dgdz[..., 0, 0] = self.f_u / z[..., 2, 0]
    dgdz[..., 0, 2] = -self.f_u * z[..., 0, 0] / (z[..., 2, 0]**2)
    dgdz[..., 1, 1] = self.f_v / z[..., 2, 0]
    dgdz[..., 1, 2] = -self.f_v * z[..., 1, 0] / (z[..., 2, 0]**2)
    dgdz[..., 2, 0] = self.f_u / z[..., 2, 0]
    dgdz[..., 2, 2] = -self.f_u * (z[..., 0, 0] - self.b) / (z[..., 2, 0]**2)
    dgdz[..., 3, 1] = self.f_v / z[..., 2, 0]
    dgdz[..., 3, 2] = -self.f_v * z[..., 1, 0] / (z[..., 2, 0]**2)
    dzdx = self.D @ self.T_cv @ se3op.point2fs(T @ p)
    return dgdz @ dzdx


if __name__ == "__main__":

  dataset = "/home/yuchen/ASRL/aer1513/AER1513-A3/code/dataset3.mat"

  # Plot valid measurements
  print('Q4 Plot valid measurements')
  estimator = Estimator(dataset)
  estimator.plot_num_visible_landmarks()

  # Batch case
  print('Q5(a) batch optimization')
  estimator = Estimator(dataset)
  estimator.set_interval(1215, 1715)
  estimator.initialize()  # initialize with odometry
  estimator.optimize()
  batch_time = estimator.optimization_time
  estimator.plot_error("batch")

  # print('Q5(b) sliding window optimization with kappa=50')
  # k1 = 1215
  # k2 = 1714
  # kappa = 50
  # estimator = Estimator(dataset)
  # estimator.set_interval(k1, k1 + 50)
  # # initialize with odometry using ground truth
  # estimator.initialize()
  # estimator.optimize()
  # for k in range(k1 + 1, k2 ):
  #   print('Current k =', k)
  #   estimator.set_interval(k, k + 50)
  #   # initialize with odometry at the previous step
  #   estimator.initialize(k - 1)
  #   estimator.optimize()
  # sliding_50_time = estimator.optimization_time
  # estimator.plot_error("sliding_window_50", k1, k2)

  # print('Q5(b) sliding window optimization with kappa=10')
  # k1 = 1215
  # k2 = 1714
  # kappa = 10
  # estimator = Estimator(dataset)
  # estimator.set_interval(k1, k1 + kappa)
  # # initialize with odometry using ground truth
  # estimator.initialize()
  # estimator.optimize()
  # for k in range(k1 + 1, k2 ):
  #   print('Current k =', k)
  #   estimator.set_interval(k, k + kappa)
  #   # initialize with odometry at the previous step
  #   estimator.initialize(k - 1)
  #   estimator.optimize()
  # sliding_10_time = estimator.optimization_time
  # estimator.plot_error("sliding_window_10", k1, k2)

  # print("Timing - ")
  # print("batch:                 ", batch_time)
  # print("sliding window k = 50: ", sliding_50_time)
  # print("sliding window k = 10: ", sliding_10_time)
