"""
nn_layers.py — simple Taichi-backed neural-network layers and optimisers.

Self-contained; no module-level Taichi field creation.  All `ti.field`
allocations happen inside class `__init__` so this module is safe to import
before or after `ti.init()` — you just can't *instantiate* the classes until
Taichi is initialised.

Classes
-------
SGD      — stochastic gradient descent with gradient clipping (±20).
AdamW    — Adam with decoupled weight decay and per-element gradient clipping.
Linear   — one fully-connected layer with optional tanh activation.
"""
import pickle as pkl
import taichi as ti

real = ti.f32
scalar = lambda: ti.field(dtype=real)


# ==============================================================================
#  Optimisers
# ==============================================================================

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
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
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
    def _step(self, w: ti.template(), m: ti.template(),
              v: ti.template(), t: ti.i32):
        for I in ti.grouped(w):
            g = w.grad[I]
            if g != g:                # nan guard
                g = 0.0
            g = ti.min(ti.max(g, -20.0), 20.0)   # gradient clipping
            m[I] = self.beta1 * m[I] + (1.0 - self.beta1) * g
            v[I] = self.beta2 * v[I] + (1.0 - self.beta2) * g * g
            m_hat = m[I] / (1.0 - ti.pow(self.beta1, t))
            v_hat = v[I] / (1.0 - ti.pow(self.beta2, t))
            w[I] -= self.lr * (m_hat / (ti.sqrt(v_hat) + self.eps)
                               + self.weight_decay * w[I])

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


# ==============================================================================
#  Layers
# ==============================================================================

@ti.data_oriented
class Linear:
    """One fully-connected layer.

    Hidden state *accumulates* across calls (like a residual integrator);
    call `clear()` between episodes to reset.
    """

    def __init__(self, n_models, batch_size, n_steps,
                 n_input, n_hidden, n_output,
                 needs_grad=False, activation=False):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = activation
        if self.n_hidden != self.n_output:
            raise ValueError(
                "Linear currently requires n_hidden == n_output; "
                "the forward kernel writes one output per hidden unit."
            )

        self.hidden = scalar()
        self.output = scalar()

        # data layout: (n_models, n_hidden, n_input) for weights
        self.batch_node = ti.root.dense(ti.i, self.n_models)
        self.n_hidden_node = self.batch_node.dense(ti.j, self.n_hidden)
        self.weights1_node = self.n_hidden_node.dense(ti.k, self.n_input)

        self.batch_node.dense(
            ti.axes(1, 2, 3),
            (self.n_steps, self.batch_size, self.n_hidden)).place(self.hidden)
        self.batch_node.dense(
            ti.axes(1, 2, 3),
            (self.n_steps, self.batch_size, self.n_output)).place(self.output)

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
        q1 = ti.sqrt(6.0 / self.n_input) * 0.01
        for model_id, i, j in ti.ndrange(self.n_models,
                                          self.n_hidden, self.n_input):
            self.weights1[model_id, i, j] = (ti.random() * 2.0 - 1.0) * q1

    @ti.kernel
    def _forward(self, t: ti.i32, inp: ti.template()):
        for m, k, i, j in ti.ndrange(self.n_models, self.batch_size,
                                      self.n_hidden, self.n_input):
            self.hidden[m, t, k, i] += self.weights1[m, i, j] * inp[m, t, k, j]
        if ti.static(self.activation):
            for m, k, i in ti.ndrange(self.n_models, self.batch_size,
                                       self.n_hidden):
                self.output[m, t, k, i] = ti.tanh(
                    self.hidden[m, t, k, i] + self.bias1[m, i])
        else:
            for m, k, i in ti.ndrange(self.n_models, self.batch_size,
                                       self.n_hidden):
                self.output[m, t, k, i] = (self.hidden[m, t, k, i]
                                            + self.bias1[m, i])

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.0
        for I in ti.grouped(self.output):
            self.output[I] = 0.0

    @ti.kernel
    def clear_io_grad(self):
        for I in ti.grouped(self.hidden):
            self.hidden.grad[I] = 0.0
        for I in ti.grouped(self.output):
            self.output.grad[I] = 0.0

    def forward(self, t, inp):
        self._forward(t, inp)

    # ---- serialisation ----
    def dump_weights(self, name="save.pkl"):
        w_val = []
        for w in self.parameters():
            w_val.append(w.to_numpy()[0])
        with open(name, "wb") as f:
            pkl.dump(w_val, f)

    def load_weights(self, name="save.pkl", model_id=0):
        with open(name, "rb") as f:
            w_val = pkl.load(f)
        self.load_weights_from_value(w_val, model_id)

    def load_weights_from_value(self, w_val, model_id=0):
        for w, val in zip(self.parameters(), w_val):
            expected_shape = w.shape[1:]
            if val.shape != expected_shape and val.shape[0] == 1:
                val = val[0]
            self.copy_from_numpy(w, val, model_id)

    @staticmethod
    @ti.kernel
    def copy_from_numpy(dst: ti.template(), src: ti.types.ndarray(),
                        model_id: ti.i32):
        for I in ti.grouped(src):
            dst[model_id, I] = src[I]