# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implicit differentiation of ridge regression.
=============================================
"""

from absl import app
import jax
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver
from matplotlib.pyplot import vlines
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing


def ridge_objective(params, l2reg, data):
  """Ridge objective function."""
  X_tr, y_tr = data
  residuals = jnp.dot(X_tr, params) - y_tr
  return 0.5 * jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.sum(params ** 2)


@implicit_diff.custom_root(jax.grad(ridge_objective))
def ridge_solver(init_params, l2reg, data):
  """Solve ridge regression by conjugate gradient."""
  X_tr, y_tr = data

  def matvec(u):
    return jnp.dot(X_tr.T, jnp.dot(X_tr, u))

  return linear_solve.solve_cg(matvec=matvec,
                               b=jnp.dot(X_tr.T, y_tr),
                               ridge=len(y_tr) * l2reg,
                               init=init_params,
                               maxiter=20)


# Perhaps confusingly, theta is a parameter of the outer objective,
# but l2reg = jnp.exp(theta) is an hyper-parameter of the inner objective.
def outer_objective(theta, init_inner, data):
  """Validation loss."""
  X_tr, X_val, y_tr, y_val = data
  # We use the bijective mapping l2reg = jnp.exp(theta)
  # both to optimize in log-space and to ensure positivity.
  l2reg = jnp.exp(theta)
  w_fit = ridge_solver(init_inner, l2reg, (X_tr, y_tr))
  y_pred = jnp.dot(X_val, w_fit)
  loss_value = jnp.mean((y_pred - y_val) ** 2)
  # We return w_fit as auxiliary data.
  # Auxiliary data is stored in the optimizer state (see below).
  return loss_value, w_fit


def optimize_hyper():
  # Prepare data.
  X, y = datasets.load_boston(return_X_y=True)
  X = preprocessing.normalize(X)
  # data = (X_tr, X_val, y_tr, y_val)
  data = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

  # Initialize solver.
  solver = OptaxSolver(opt=optax.adam(1e-2), fun=outer_objective, has_aux=True)
  theta = 1.0
  state = solver.init_state(theta)
  init_w = jnp.zeros(X.shape[1])

  # Run outer loop.
  for _ in range(50):
    theta, state = solver.update(params=theta, state=state, init_inner=init_w,
                                 data=data)
    # The auxiliary data returned by the outer loss is stored in the state.
    init_w = state.aux
    print(f"[Step {state.iter_num}] Validation loss: {state.value:.3f}.")

def plot_tangent():
    X, y = datasets.load_boston(return_X_y=True)
    X = preprocessing.normalize(X)
    # data = (X_tr, X_val, y_tr, y_val)
    data = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)
    init_w = jnp.zeros(X.shape[1])
    count = 20
    thetas = jnp.linspace(0,2,count)
    delta = 0.001
    valid_loss = jnp.array([outer_objective(thetas[i], init_w, data)[0] for i in range(count)])
    grad_thetas = jnp.array([jax.jacobian(outer_objective,argnums=0)(thetas[i], init_w, data)[0] for i in range(count)])
    fd_thetas = jnp.array([(outer_objective(thetas[i] + delta/2, init_w, data)[0] - outer_objective(thetas[i] - delta/2, init_w, data)[0]) / delta for i in range(count)])
    import matplotlib.pylab as plt
    import tqdm
    plt.figure(figsize=(24,7))
    plt.subplot(1,3,1)
    plt.plot(thetas,valid_loss,'r')
    for i in tqdm.trange(valid_loss.shape[0]):
        plt.arrow(thetas[i],valid_loss[i],1,grad_thetas[i],color='g')
    plt.subplot(1,3,2)
    plt.plot(thetas,valid_loss,'r')
    for i in tqdm.trange(valid_loss.shape[0]):
        plt.arrow(thetas[i],valid_loss[i],1,fd_thetas[i],color='b')
    plt.subplot(1,3,3)
    plt.plot(thetas,valid_loss,'r')
    for i in tqdm.trange(valid_loss.shape[0]):
        plt.arrow(thetas[i],valid_loss[i],1,grad_thetas[i],color='g')
        plt.arrow(thetas[i],valid_loss[i],1,fd_thetas[i],color='b')
        
    plt.savefig('out/plot.pdf')
    plt.close()
def main(argv):
  del argv
  plot_tangent()



if __name__ == "__main__":
  app.run(main)