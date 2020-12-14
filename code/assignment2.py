import os
import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse

# Configure matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc("font", size=MEDIUM_SIZE)        # controls default text sizes
plt.rc("figure", titlesize=MEDIUM_SIZE) # fontsize of the figure title
plt.rc("axes", titlesize=MEDIUM_SIZE)   # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)   # legend fontsize

# Helpers
wrap_to_pi = lambda th: (th + np.pi) % (2 * np.pi) - np.pi

# Load data
data = loadmat('dataset2.mat')
start_id, stop_id = 0, None
K = data["t"].shape[0]
for k, v in data.items():
  try:
    if v.shape[0] == K:
      data[k] = v[start_id:stop_id]
  except:
    pass

# Problem specific parameters
K = data["x_true"].shape[0] # number of timesteps
T = 0.1                     # sampling period
d = data["d"][0, 0]         # distance b/t robot center & laser range finder
rmax = 10                   # maximum range (will be changed later)

# Problem specific vectors and matrices (refer to Barfoot, 2017 Chapter 4.2.3)
Q = np.array([[data["v_var"][0, 0], 0.], [0., data["om_var"][0, 0]]])

Rl = np.array([[data["r_var"][0, 0], 0.], [0., data["b_var"][0, 0]]])
R = lambda l: block_diag(*[Rl for k in range(int(l.shape[0] / 2))])

f = lambda x, v: x + T * np.array([
    [np.cos(x[2, 0]), 0],
    [np.sin(x[2, 0]), 0],
    [0, 1],
]) @ v

F = lambda x, v: np.array([
    [1, 0, -T * np.sin(x[2, 0]) * v[0, 0]],
    [0, 1, T * np.cos(x[2, 0]) * v[0, 0]],
    [0, 0, 1],
])

W = lambda x: T * np.array([
    [np.cos(x[2, 0]), 0],
    [np.sin(x[2, 0]), 0],
    [0, 1],
])

dx = lambda x, l: l[0, 0] - x[0, 0] - d * np.cos(x[2, 0])
dy = lambda x, l: l[1, 0] - x[1, 0] - d * np.sin(x[2, 0])

gl = lambda x, l: np.array([
    [np.sqrt(dx(x, l)**2 + dy(x, l)**2)],
    [np.arctan2(dy(x, l), dx(x, l)) - x[2, 0]],
])
g = lambda x, l: np.concatenate([gl(x, l[2 * k:2 * k + 2]) for k in range(int(l.shape[0] / 2))], axis=0)

dgldx = lambda x, l: np.array([
    [-(dx(x, l)**2 + dy(x, l)**2)**(-1 / 2) * dx(x, l)],
    [dy(x, l) / (dx(x, l)**2 + dy(x, l)**2)],
])
dgldy = lambda x, l: np.array([
    [-(dx(x, l)**2 + dy(x, l)**2)**(-1 / 2) * dy(x, l)],
    [-dx(x, l) / (dx(x, l)**2 + dy(x, l)**2)],
])
dgldth = lambda x, l: np.array([
    [(dx(x, l)**2 + dy(x, l)**2)**(-1 / 2) * (dx(x, l) * d * np.sin(x[2, 0]) - dy(x, l) * d * np.cos(x[2, 0]))],
    [-(dx(x, l) * d * np.cos(x[2, 0]) + dy(x, l) * d * np.sin(x[2, 0])) / (dx(x, l)**2 + dy(x, l)**2) - 1],
])
Gl = lambda x, l: np.concatenate((dgldx(x, l), dgldy(x, l), dgldth(x, l)), axis=1)
G = lambda x, l: np.concatenate([Gl(x, l[2 * k:2 * k + 2]) for k in range(int(l.shape[0] / 2))], axis=0)

Nl = lambda _=None: np.eye(2)
N = lambda x, l: block_diag(*[Nl() for k in range(int(l.shape[0] / 2))])


def ekf(x0, P0, vs, rs, bs, lall, x_trues=np.array([])):
  global K, rmax

  x_hats = [x0]
  P_hats = [P0]

  x_hat = x0
  P_hat = P0
  for k in range(1, K):
    # terminal output
    print("EKF: step {} - {:4.2f}%".format(k, k / K * 100), end='\r')

    # control input
    v = vs[k:k + 1, :].T

    # filter observations according to rmax
    r = rs[k, :]
    b = bs[k, :]
    obs_idx = np.argwhere(np.logical_and(r > 0, r < rmax)).reshape(-1)
    has_obs = len(obs_idx) > 0
    r = r[obs_idx, None]
    b = b[obs_idx, None]
    y = np.concatenate((r, b), axis=1).reshape((-1, 1))
    l = lall[obs_idx].reshape((-1, 1))

    # predictor
    x_op = x_hat if x_trues.size == 0 else x_trues[k - 1] # x_op is at k-1
    x_op[2, 0] = wrap_to_pi(x_op[2, 0])
    x_hat[2, 0] = wrap_to_pi(x_hat[2,0])

    P_check = F(x_op, v) @ P_hat @ F(x_op, v).T + W(x_op) @ Q @ W(x_op).T
    P_check = 0.5 * (P_check + P_check.T)

    x_check = f(x_hat, v)
    x_check[2, 0] = wrap_to_pi(x_check[2, 0])

    if has_obs:
      x_op = x_check if x_trues.size == 0 else x_trues[k]
      x_op[2, 0] = wrap_to_pi(x_op[2, 0])

      # Kalman gain
      Km = P_check @ G(x_op, l).T @ inv(G(x_op, l) @ P_check @ G(x_op, l).T + N(x_op, l) @ R(l) @ N(x_op, l).T)

      # corrector
      P_hat = (np.eye(x_op.shape[0]) - Km @ G(x_op, l)) @ P_check
      P_hat = 0.5 * (P_hat + P_hat.T)

      obs_err = y - g(x_check, l)
      obs_err[range(1, len(l), 2)] = wrap_to_pi(obs_err[range(1, len(l), 2)])
      x_hat = x_check + Km @ obs_err
    else:
      P_hat = P_check
      x_hat = x_check
    x_hat[2, 0] = wrap_to_pi(x_hat[2, 0])

    # store results
    x_hats.append(x_hat)
    P_hats.append(P_hat)

  print("EKF: step {} - 100.0% - done!".format(K))
  return np.array(x_hats), np.array(P_hats)


# true states [x, y, om]
x_trues = np.concatenate((data["x_true"], data["y_true"], wrap_to_pi(data["th_true"])), axis=1)[..., None]
# true landmark locations
l = data["l"]
# input sequence
vs = np.concatenate((data["v"], data["om"]), axis=1)
# observations
rs = data["r"]
bs = data["b"]


def plot_q4(x, x_true, P, t, id):
  global num_figure, K

  for ts in [0, 1000]:
    if ts > K:
      continue
    num_figure += 1

    dx = (x - x_true).squeeze()
    dx[:, 2] = wrap_to_pi(dx[:, 2])
    stdx = np.array([np.sqrt(np.diag(p)) for p in P])

    fig = plt.figure(num_figure)
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.1, top=0.95, wspace=0.6, hspace=0.6)

    plt.subplot(311)
    plt.plot(t[ts:], dx[ts:, 0], linewidth=1.0)
    plt.plot(t[ts:], -3 * stdx[ts:, 0], "r--", linewidth=1.0)
    plt.plot(t[ts:], +3 * stdx[ts:, 0], "g--", linewidth=1.0)
    plt.fill_between(t[ts:], -3 * stdx[ts:, 0], 3 * stdx[ts:, 0], alpha=0.2)
    plt.xlabel(r"$t$ [$s$]")
    plt.ylabel(r"$\hat{x} - x$ [$m$]")
    if ts != 0:
      plt.ylim([-0.2, 0.2])
    print("Starting time: ", ts, " average dx: ", np.linalg.norm(dx[ts:, 0], ord=2)**2)

    plt.subplot(312)
    plt.plot(t[ts:], dx[ts:, 1], linewidth=1.0)
    plt.plot(t[ts:], -3 * stdx[ts:, 1], "r--", linewidth=1.0)
    plt.plot(t[ts:], +3 * stdx[ts:, 1], "g--", linewidth=1.0)
    plt.fill_between(t[ts:], -3 * stdx[ts:, 1], 3 * stdx[ts:, 1], alpha=0.2)
    plt.xlabel(r"$t$ [$s$]")
    plt.ylabel(r"$\hat{y}-y$ [$m$]")
    if ts != 0:
      plt.ylim([-0.2, 0.2])
    print("Starting time: ", ts, " average dy: ", np.linalg.norm(dx[ts:, 1], ord=2)**2)

    plt.subplot(313)
    plt.plot(t[ts:], dx[ts:, 2], linewidth=1.0)
    plt.plot(t[ts:], -3 * stdx[ts:, 2], "r--", linewidth=1.0)
    plt.plot(t[ts:], +3 * stdx[ts:, 2], "g--", linewidth=1.0)
    plt.fill_between(t[ts:], -3 * stdx[ts:, 2], 3 * stdx[ts:, 2], alpha=0.2)
    plt.xlabel(r"$t$ [$s$]")
    plt.ylabel(r"$\hat{\theta}-\theta$ [$rad$]")
    if ts != 0:
      plt.ylim([-0.2, 0.2])
    print("Starting time: ", ts, " average dth: ", np.linalg.norm(dx[ts:, 2], ord=2)**2)

    os.makedirs('figures', exist_ok=True)
    plt.savefig("figures/q4-{}-{}.png".format(id, ts))


num_figure = 0

print("Q4(a)...")
for rmax in [5, 3, 1]:
  x0_hat = np.concatenate((data["x_true"][:1], data["y_true"][:1], wrap_to_pi(data["th_true"][:1])), axis=1).T
  P0_hat = np.diag((1, 1, 0.1))
  x_hats, P_hats = ekf(x0_hat, P0_hat, vs, rs, bs, l)
  plot_q4(x_hats, x_trues, P_hats, data["t"][:, 0], "a-" + str(rmax))

print("Q4(b)...")
for rmax in [5, 3, 1]:
  x0_hat = np.array([[1.], [1.], [0.1]])
  P0_hat = np.diag((1, 1, 0.1))
  x_hats, P_hats = ekf(x0_hat, P0_hat, vs, rs, bs, l)
  plot_q4(x_hats, x_trues, P_hats, data["t"][:, 0], "b-" + str(rmax))

print("Q4(b)'...")  # a different initial guess
for rmax in [5, 3, 1]:
  x0_hat = np.array([[100.], [100.], [2.0]])
  P0_hat = np.diag((1, 1, 0.1))
  x_hats, P_hats = ekf(x0_hat, P0_hat, vs, rs, bs, l)
  plot_q4(x_hats, x_trues, P_hats, data["t"][:, 0], "b2-" + str(rmax))

print("Q4(c)...")
for rmax in [5, 3, 1]:
  x0_hat = np.concatenate((data["x_true"][:1], data["y_true"][:1], wrap_to_pi(data["th_true"][:1])), axis=1).T
  P0_hat = np.diag((1, 1, 0.1))
  x_hats, P_hats = ekf(x0_hat, P0_hat, vs, rs, bs, l, x_trues=x_trues)
  plot_q4(x_hats, x_trues, P_hats, data["t"][:, 0], "c-" + str(rmax))

print("Q5...")
num_figure += 1

rmax = 1
x0_hat = np.concatenate((data["x_true"][:1], data["y_true"][:1], wrap_to_pi(data["th_true"][:1])), axis=1).T
P0_hat = np.diag((1, 1, 0.1))
x_hats, P_hats = ekf(x0_hat, P0_hat, vs, rs, bs, l)

x_trues = x_trues.squeeze()
x_hats = x_hats.squeeze()

fig = plt.figure(num_figure)
ax = plt.axes(aspect=1)
plt.title(r"EKF estimation with $r_{max}=1$, $\hat{x}_0=x_0$ and $\hat{P}_0=$diag$\{1, 1, 0.1\}$")
plt.xlabel(r"$x$ [$m$]")
plt.ylabel(r"$y$ [$m$]")
plt.xlim([-2, 10])
plt.ylim([-4, 4])
plt.scatter(data["l"][:, 0], data["l"][:, 1], c="k", s=3)
x_true_p = plt.scatter([], [], c="b", s=3)
x_hat_p = plt.scatter([], [], c="r", s=3)

cov = P_hats[0, :2, :2]
lmda, v = np.linalg.eig(cov)
lmda = np.sqrt(lmda)

ell = Ellipse(xy=x_hats[0, :2], width=3 * lmda[0] * 2, height=3 * lmda[1] * 2, angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])), fc="r", ec="r", alpha=0.2)


def init():
  x_true_p.set_offsets(x_trues[0:1, :2])
  x_hat_p.set_offsets(x_hats[0:1, :2])
  ax.add_artist(ell)
  return x_true_p, x_hat_p, ell


def animate(i, *args):
  x_true_p.set_offsets(x_trues[i:i + 1, :2])
  x_hat_p.set_offsets(x_hats[i:i + 1, :2])

  cov = P_hats[i, :2, :2]
  lmda, v = np.linalg.eig(cov)
  lmda = np.sqrt(lmda)
  ell.set_center(x_hats[i, :2])
  ell.set_width(3 * lmda[0] * 2)
  ell.set_height(3 * lmda[1] * 2)
  ell.set_angle(np.rad2deg(np.arctan2(v[1, 0], v[0, 0])))
  return x_true_p, x_hat_p, ell


anim = FuncAnimation(fig, animate, init_func=init, frames=K, interval=5, blit=True)
os.makedirs('figures', exist_ok=True)
writervideo = FFMpegWriter(fps=60)
anim.save("figures/AER1513_A2.mov", writer=writervideo)

# plt.show()
print("Finished!")