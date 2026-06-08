import taichi as ti

ti.init(print_ir=True)
dim = 2
F_1 = ti.Matrix.field(dim,
                      dim,
                      dtype=ti.f32,
                      shape=1,
                      needs_grad=True)
F = ti.Matrix.field(dim,
                    dim,
                    dtype=ti.f32,
                    shape=1,
                    needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def init():
    F[0] = ti.Matrix([[1.0, 0.5],
                       [0.3, 1.2]])


@ti.kernel
def polar():
    r, s = ti.polar_decompose(F[0])
    F_1[0] = r
    loss[None] = (F_1[0] - ti.Matrix.identity(float, 2)).norm()


init()
with ti.ad.Tape(loss=loss):
    polar()
