# This code file aims to test the availability of taichi-base NN .


import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="whether train model, default false")
parser.add_argument("place_holder", nargs="*")
args = parser.parse_args()

TRAIN = args.train
TRAIN_OUTPUT_IMG = False
TRAIN_VISUAL = False
TRAIN_VISUAL_SHOW = False
INFER_OUTPUT_IMG = False
arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch, device_memory_fraction=0.5, random_seed=5)
screen_res = (800, 800)

dtype_f_np = np.float32
real = ti.f32
scalar = lambda: ti.field(dtype=real)


@ti.data_oriented
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for w in self.params:
            self._step(w)

    @ti.kernel
    def _step(self, w: ti.template()):
        for I in ti.grouped(w):
            w[I] -= ti.min(ti.max(w.grad[I], -20.0), 20.0) * self.lr

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = []
        self.v = []
        for w in params:
            self.m.append(ti.field(dtype=real, shape=w.shape))
            self.v.append(ti.field(dtype=real, shape=w.shape))
            self.m[-1].fill(0.0)
            self.v[-1].fill(0.0)

        self.t = ti.field(dtype=ti.i32, shape=())

    def step(self):
        self.t[None] += 1
        for i, w in enumerate(self.params):
            self._step(w, self.m[i], self.v[i], self.t[None])

    @ti.kernel
    def _step(self, w: ti.template(), m: ti.template(), v: ti.template(), t: ti.i32):
        for I in ti.grouped(w):
            g = w.grad[I]
            m[I] = self.beta1 * m[I] + (1.0 - self.beta1) * g
            v[I] = self.beta2 * v[I] + (1.0 - self.beta2) * g * g
            m_hat = m[I] / (1.0 - ti.pow(self.beta1, t))
            v_hat = v[I] / (1.0 - ti.pow(self.beta2, t))
            w[I] -= self.lr * (m_hat / (ti.sqrt(v_hat) + self.eps) + self.weight_decay * w[I])

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class Linear:
    def __init__(
        self,
        n_models,
        batch_size,
        n_steps,
        n_input,
        n_hidden,
        n_output,
        needs_grad=False,
        activation=False,
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = activation

        self.hidden = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)
        self.n_hidden_node = self.batch_node.dense(ti.j, self.n_hidden)
        self.weights1_node = self.n_hidden_node.dense(ti.k, self.n_input)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_hidden)).place(self.hidden)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights1_node.place(self.weights1)
        self.n_hidden_node.place(self.bias1)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.weights1, self.bias1]

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / self.n_input) * 0.01
        for model_id, i, j in ti.ndrange(self.n_models, self.n_hidden, self.n_input):
            self.weights1[model_id, i, j] = (ti.random() * 2 - 1) * q1

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input):
            self.hidden[model_id, t, k, i] += self.weights1[model_id, i, j] * nn_input[model_id, t, k, j]
        if ti.static(self.activation):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = ti.tanh(self.hidden[model_id, t, k, i] + self.bias1[model_id, i])
        else:
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = self.hidden[model_id, t, k, i] + self.bias1[model_id, i]

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.0
        for I in ti.grouped(self.output):
            self.output[I] = 0.0

    def forward(self, t, nn_input):
        self._forward(t, nn_input)

    def dump_weights(self, name="save.pkl"):
        w_val = []
        for w in self.parameters():
            w = w.to_numpy()
            w_val.append(w[0])
        with open(name, "wb") as f:
            pkl.dump(w_val, f)

    def load_weights(self, name="save.pkl", model_id=0):
        with open(name, "rb") as f:
            w_val = pkl.load(f)
        self.load_weights_from_value(w_val, model_id)

    def load_weights_from_value(self, w_val, model_id=0):
        for w, val in zip(self.parameters(), w_val):
            if val.shape[0] == 1:
                val = val[0]
            self.copy_from_numpy(w, val, model_id)

    @staticmethod
    @ti.kernel
    def copy_from_numpy(dst: ti.template(), src: ti.types.ndarray(), model_id: ti.i32):
        for I in ti.grouped(src):
            dst[model_id, I] = src[I]


@ti.data_oriented
class Model:
    def __init__(self, n_models, batch_size, n_steps, layer_dims, needs_grad=False):
        self.layers = []
        n_hidden = max(d[1] for d in layer_dims)
        for in_dim, out_dim, activation in layer_dims:
            self.layers.append(
                Linear(
                    n_models=n_models,
                    batch_size=batch_size,
                    n_steps=n_steps,
                    n_input=in_dim,
                    n_hidden=n_hidden,
                    n_output=out_dim,
                    needs_grad=needs_grad,
                    activation=activation,
                )
            )

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def weights_init(self):
        for layer in self.layers:
            layer.weights_init()

    def forward(self, t, nn_input):
        x = nn_input
        for layer in self.layers:
            layer.forward(t, x)
            x = layer.output
        return x

    def clear(self):
        for layer in self.layers:
            layer.clear()

    def dump_weights(self, path_prefix="model"):
        for i, layer in enumerate(self.layers):
            layer.dump_weights(f"{path_prefix}_layer{i}.pkl")

    def load_weights(self, path_prefix="model", model_id=0):
        for i, layer in enumerate(self.layers):
            layer.load_weights(f"{path_prefix}_layer{i}.pkl", model_id)


@ti.data_oriented
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss = ti.field(dtype=real, shape=(), needs_grad=True)

    def train_step(self, forward_fn):
        self.model.clear()
        self.loss[None] = 0.0
        with ti.ad.Tape(loss=self.loss):
            forward_fn()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.loss[None]

    @ti.kernel
    def compute_mse_loss(self, pred: ti.template(), target: ti.template()):
        for I in ti.grouped(pred):
            self.loss[None] += (pred[I] - target[I]) ** 2

    @ti.kernel
    def compute_mae_loss(self, pred: ti.template(), target: ti.template()):
        for I in ti.grouped(pred):
            self.loss[None] += ti.abs(pred[I] - target[I])