# BPTT 为何必须保留全部中间状态：数学推导

## 1. 问题设定

将 SPH + 神经控制器系统抽象为离散动力系统。

### 状态变量

记第 $t$ 步的物理状态为向量：

$$
\mathbf{S}_t =
\begin{bmatrix}
\mathbf{x}_t \\ \mathbf{v}_t
\end{bmatrix}
\in \mathbb{R}^{2 \cdot 3P}
$$

其中 $\mathbf{x}_t = (\mathbf{r}_{t,1}, \dots, \mathbf{r}_{t,P})$ 为 $P$ 个粒子的三维位置，$\mathbf{v}_t$ 同理为速度。

### 状态转移

系统演化方程（对应代码中 `advance`, `update_force`, `apply_force` 的合效果）：

$$
\mathbf{S}_t = \Phi\left(\mathbf{S}_{t-1},\; \mathbf{u}_{t-1}(\boldsymbol{\theta})\right), \qquad t = 1, 2, \dots, T
$$

其中：
- $\mathbf{u}_t(\boldsymbol{\theta}) \in \mathbb{R}^3$ 为神经控制器在第 $t$ 步输出的喷射力（即 `F_jet_force[t, :]`）；
- $\boldsymbol{\theta}$ 为神经网络的全部可学习参数（$\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2$）；
- $\Phi$ 是确定性映射，包含了密度/压力更新、力计算、速度/位置积分、边界处理等全部子步。

### 损失函数

损失仅在最后一步 $T$ 计算（`compute_loss(steps-1)` 处）：

$$
\mathcal{L} = \ell(\mathbf{S}_T)
$$

在代码中，$\ell$ 根据流体粒子与目标中心的最小距离和最大散布来度量喷泉命中精度。

### 梯度目标

训练需要计算参数梯度，以便执行梯度下降（`optimizer.step()`）：

$$
\frac{d\mathcal{L}}{d\boldsymbol{\theta}} \in \mathbb{R}^{|\boldsymbol{\theta}|}
$$

**下面证明：此梯度的计算需要全部中间状态 $\mathbf{S}_0, \mathbf{S}_1, \dots, \mathbf{S}_{T-1}$ 的前向值。**

---

## 2. 梯度沿时间展开

由于 $\boldsymbol{\theta}$ 通过**每一步**的控制器输出 $\mathbf{u}_t(\boldsymbol{\theta})$ 影响最终损失，将 $\frac{d\mathcal{L}}{d\boldsymbol{\theta}}$ 按时间拆分为 $T$ 个贡献项。

### 2.1 标量参数情形的链式法则

先假设 $\theta$ 为标量（仅一个参数）。由全导数公式：

$$
\frac{d\mathcal{L}}{d\theta}
= \frac{\partial \ell}{\partial \mathbf{S}_T} \cdot \frac{d\mathbf{S}_T}{d\theta}
$$

对 $\frac{d\mathbf{S}_T}{d\theta}$ 递归展开一次：

$$
\frac{d\mathbf{S}_T}{d\theta}
= \frac{\partial \Phi}{\partial \mathbf{S}_{T-1}} \cdot \frac{d\mathbf{S}_{T-1}}{d\theta}
+ \frac{\partial \Phi}{\partial \mathbf{u}_{T-1}} \cdot \frac{d\mathbf{u}_{T-1}}{d\theta}
$$

将 $\frac{d\mathbf{S}_{T-1}}{d\theta}$ 继续展开，直到 $\mathbf{S}_0$（$\mathbf{S}_0$ 由 `initialize_fluid_particle` 初始化，与 $\theta$ 无关，即 $\frac{d\mathbf{S}_0}{d\theta}=0$）：

$$
\begin{aligned}
\frac{d\mathcal{L}}{d\theta}
&= \frac{\partial \ell}{\partial \mathbf{S}_T}
\cdot \frac{\partial \Phi}{\partial \mathbf{u}_{T-1}}
\cdot \frac{d\mathbf{u}_{T-1}}{d\theta}
\\[4pt]
&+ \frac{\partial \ell}{\partial \mathbf{S}_T}
\cdot \frac{\partial \Phi}{\partial \mathbf{S}_{T-1}}
\cdot \frac{\partial \Phi}{\partial \mathbf{u}_{T-2}}
\cdot \frac{d\mathbf{u}_{T-2}}{d\theta}
\\[4pt]
&+ \frac{\partial \ell}{\partial \mathbf{S}_T}
\cdot \frac{\partial \Phi}{\partial \mathbf{S}_{T-1}}
\cdot \frac{\partial \Phi}{\partial \mathbf{S}_{T-2}}
\cdot \frac{\partial \Phi}{\partial \mathbf{u}_{T-3}}
\cdot \frac{d\mathbf{u}_{T-3}}{d\theta}
\\[4pt]
&+ \;\cdots
\\[4pt]
&+ \frac{\partial \ell}{\partial \mathbf{S}_T}
\cdot \frac{\partial \Phi}{\partial \mathbf{S}_{T-1}} \cdot \frac{\partial \Phi}{\partial \mathbf{S}_{T-2}} \cdot \;\cdots\; \cdot \frac{\partial \Phi}{\partial \mathbf{S}_{1}}
\cdot \frac{\partial \Phi}{\partial \mathbf{u}_{0}}
\cdot \frac{d\mathbf{u}_{0}}{d\theta}
\end{aligned}
$$

观察规律：第 $t$ 项的动力学雅可比连乘总是按 **$\tau$ 从 $T-1$ 递减到 $t+1$** 的顺序——即

$$
\frac{\partial \Phi}{\partial \mathbf{S}_{T-1}} \cdot
\frac{\partial \Phi}{\partial \mathbf{S}_{T-2}} \cdot \;\cdots\; \cdot
\frac{\partial \Phi}{\partial \mathbf{S}_{t+1}}
$$

写成紧凑求和形式（乘积按 $\tau = T-1, T-2, \dots, t+1$ 降序）：

$$
\boxed{
\frac{d\mathcal{L}}{d\theta}
= \sum_{t=0}^{T-1}
\left[
\frac{\partial \ell}{\partial \mathbf{S}_T}
\cdot \left( \prod_{\tau = T-1}^{t+1} \frac{\partial \Phi}{\partial \mathbf{S}_{\tau}} \right)
\cdot \frac{\partial \Phi}{\partial \mathbf{u}_{t}}
\right]
\cdot \frac{d\mathbf{u}_{t}}{d\theta}
}
$$

其中约定当 $t = T-1$ 时连乘积为空，退化为单位矩阵 $\mathbf{I}$（最后一步无动力学雅可比穿越）。

### 2.2 矢量参数推广

对于参数向量 $\boldsymbol{\theta} = (\theta_1, \dots, \theta_D)$，每个分量 $\theta_k$ 有相同形式的展开。写成矩阵形式：

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}
= \sum_{t=0}^{T-1}
\left(
\frac{\partial \ell}{\partial \mathbf{S}_T}
\cdot \left( \prod_{\tau = T-1}^{t+1} \frac{\partial \Phi}{\partial \mathbf{S}_{\tau}} \right)
\cdot \frac{\partial \Phi}{\partial \mathbf{u}_{t}}
\right)
\cdot \mathbf{J}_{\boldsymbol{\theta}}(\mathbf{u}_t)
$$

其中 $\mathbf{J}_{\boldsymbol{\theta}}(\mathbf{u}_t) = \frac{\partial \mathbf{u}_t}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{3 \times D}$ 是控制器输出对参数的雅可比矩阵。

---

## 3. 核心论证：为何需要全部中间状态

### 3.1 第 $t$ 项梯度需要 $\mathbf{S}_t$

梯度贡献的第 $t$ 项中，动力学雅可比连乘因子为：

$$
\frac{\partial \Phi}{\partial \mathbf{S}_{\tau}}(\mathbf{S}_{\tau}, \mathbf{u}_{\tau}), \quad \tau = t+1, \dots, T-1
$$

**每一个** $\frac{\partial \Phi}{\partial \mathbf{S}_{\tau}}$ **都是在前向值 $\mathbf{S}_{\tau}$ 处求值的一个矩阵。** 这不是常量——不同 $\tau$ 的粒子构型不同，雅可比也不同。

举个具体例子。在 `update_force` kernel 中，粒子 $i$ 受到的压力梯度力为（代码 line 448–456）：

> ```python
> F_acc[t, i] += -mass
>     * (pre[t,i]/den[t,i]² + pre[t,j]/den[t,j]²)
>     * W_gradient(pos[t,i] - pos[t,j], H)
> ```

记 $\mathbf{r}_{t,i} = \mathbf{x}_t[i]$（粒子 $i$ 在时刻 $t$ 的位置）。该项对 $\mathbf{r}_{t,j}$ 的偏导数**依赖于 $\mathbf{r}_{t,i}$ 和 $\mathbf{r}_{t,j}$ 的值**：

$$
\frac{\partial (\text{pressure\_force}_{t,i})}{\partial \mathbf{r}_{t,j}}
= f\big(\mathbf{r}_{t,i},\; \mathbf{r}_{t,j},\; \rho_{t,i},\; \rho_{t,j},\; p_{t,i},\; p_{t,j}\big)
$$

若前向值 $\mathbf{r}_{t,i}$ 已丢失，此雅可比矩阵**不可计算**。

### 3.2 不同 $t$ 的项需要不同 $\tau$ 的前向值

- $t=0$ 的贡献项需要 $\frac{\partial \Phi}{\partial \mathbf{S}_{1}}, \frac{\partial \Phi}{\partial \mathbf{S}_{2}}, \dots, \frac{\partial \Phi}{\partial \mathbf{S}_{T-1}}$  → 需要 $\mathbf{S}_1, \mathbf{S}_2, \dots, \mathbf{S}_{T-1}$
- $t=T-2$ 的贡献项仅需要 $\frac{\partial \Phi}{\partial \mathbf{S}_{T-1}}$ → 仅需 $\mathbf{S}_{T-1}$

**不同的 $t$ 对 "需要哪些 $\mathbf{S}_{\tau}$" 是互补而非重叠的。** 要完整计算 $\nabla_{\boldsymbol{\theta}} \mathcal{L}$，就必须能访问 $\mathbf{S}_1, \dots, \mathbf{S}_{T-1}$ **全部值**。

### 3.3 为什么 O(1) 内存不够——滚动窗口悖论

假设只保留最近两步 $\mathbf{S}_{t-1}, \mathbf{S}_t$ 并逐时间步滚动覆盖：

```
前向过程中:
  写入 S₁ → 内存: [S₀, S₁]
  写入 S₂ → 内存: [S₁, S₂]   (S₀ 被覆盖)
  写入 S₃ → 内存: [S₂, S₃]   (S₁ 被覆盖)
  ...
  写入 S_T → 内存: [S_{T-1}, S_T]   (S₀,...,S_{T-2} 全已丢失)

反向传播到达时间步 τ 时:
  需要 ∂Φ/∂S_τ → 需要 S_τ 的前向值 → 已丢失 → 梯度计算中断
```

**滚动窗口使前向 O(1)，但反向不可行。**

---

## 4. 标量示例

考虑一个极简的动力系统，使推导完全显式：

$$
\begin{aligned}
x_1 &= a x_0 + b \\
x_2 &= a x_1 + b \\
\mathcal{L} &= x_2
\end{aligned}
$$

参数 $\theta = (a, b)$。初始 $x_0$ 与参数无关。

### 直接展开验证

$$
x_2 = a(a x_0 + b) + b = a^2 x_0 + ab + b
$$

$$
\frac{\partial \mathcal{L}}{\partial a} = 2a x_0 + b
$$

### BPTT 链式法则验证

使用第 2 节的公式：

$$
\frac{d\mathcal{L}}{da}
= \underbrace{\frac{\partial \mathcal{L}}{\partial x_2} \cdot \frac{\partial x_2}{\partial a}}_{t=1}
+ \underbrace{\frac{\partial \mathcal{L}}{\partial x_2} \cdot \frac{\partial x_2}{\partial x_1} \cdot \frac{\partial x_1}{\partial a}}_{t=0}
$$

计算各因子：
- $\frac{\partial \mathcal{L}}{\partial x_2} = 1$
- $\frac{\partial x_2}{\partial a} = x_1 = a x_0 + b$
- $\frac{\partial x_2}{\partial x_1} = a$
- $\frac{\partial x_1}{\partial a} = x_0$

代入：
$$
\frac{d\mathcal{L}}{da}
= 1 \cdot (a x_0 + b) + 1 \cdot a \cdot x_0
= a x_0 + b + a x_0
= 2a x_0 + b \quad \checkmark
$$

### 需要哪些前向值？

| 梯度因子 | 求值处的前向值 |
|----------|---------------|
| $\frac{\partial x_2}{\partial a} = x_1$ | 需要 $x_1$ |
| $\frac{\partial x_2}{\partial x_1} = a$ | 不需要前向值（此处恰为常数 $a$） |
| $\frac{\partial x_1}{\partial a} = x_0$ | 需要 $x_0$ |

若只保留 $x_1$（覆盖 $x_0$），则 $\frac{\partial x_1}{\partial a}$ 无法计算 → 梯度错误。

**这就解释了为何必须全量存储：即使是最简单的两步线性系统，也需要 $x_0$ 和 $x_1$ 同时可用。**

---

## 5. 与本代码的直接对应

| 数学符号 | 代码中的实现 | 维度 |
|----------|-------------|------|
| $T$ | `steps = 128` | — |
| $\mathbf{S}_t$ | `F_pos[:, t, :, :]`, `F_vel[:, t, :, :]`, `den[:, t, :]`, `pre[:, t, :]` | 各 (B, P) 或 (B, P, 3) |
| $\mathbf{u}_t(\boldsymbol{\theta})$ | `F_jet_force[t, :, :]` = `fc2.output[0, t, :, :] * jet_force_max` | (B, 3) |
| $\Phi$ | `advance` + `boundary_handle` + `update_density` + `update_pressure` + `update_force` + `apply_force` | — |
| $\ell$ | `compute_loss(T-1)` → `loss += (min_dist + 0.2*max_dist) / BATCH_SIZE` | 标量 |
| $\boldsymbol{\theta}$ | `fc1.weights1`, `fc1.bias1`, `fc2.weights1`, `fc2.bias1` | — |
| 前向全量存储 | `ti.root.dense(ti.ijk, (B, steps, P)).place(F_pos, F_vel, F_acc, den, pre)` | (B, 128, P) |
| 自动微分 | `ti.ad.Tape(loss=loss):` 包裹整个循环 | — |

---

## 6. 内存替代方案：梯度检查点

全量存储并非唯一的正确方案。**梯度检查点（gradient checkpointing / memory-efficient BPTT）** 可将空间从 $O(T)$ 降至 $O(\sqrt{T})$ 或 $O(\log T)$，代价是增加前向重计算：

- **策略**：仅存储每隔 $K$ 步的快照 $\mathbf{S}_{0}, \mathbf{S}_{K}, \mathbf{S}_{2K}, \dots$
- **反向时**：需要 $\mathbf{S}_{\tau}$ 时，从最近的快照 $\mathbf{S}_{\lfloor \tau/K \rfloor K}$ 出发重新前向 $(\tau \bmod K)$ 步
- **内存复杂度**：$O(T/K)$ 而非 $O(T)$
- **计算复杂度**：前向总步数约 $2T$（原始 $T$ + 重计算 $\approx T$）

**本法未采用检查点**，因为 `ti.ad.Tape` 当前实现为最简单的全量记录策略，适合研究原型但对长序列内存不友好。128 步是 $T$ 的内存可行上界：所有粒子的全部时间切片可同时放入 GPU 显存。

---

## 7. 总结

1. **梯度 = 沿时间的求和**：每步的控制器输出 $\mathbf{u}_t(\boldsymbol{\theta})$ 都对最终 $\mathcal{L}$ 有梯度贡献，通过链式法则沿动力学状态链回传；
2. **每个动力学雅可比 $\partial \Phi / \partial \mathbf{S}_{\tau}$ 必须在其前向值 $\mathbf{S}_{\tau}$ 处求值**——这些矩阵不是常量，随粒子构型变化；
3. **丢失任一 $\mathbf{S}_{\tau}$ 就切断 $t < \tau$ 所有项的梯度链**，令低时间步的梯度贡献不可计算；
4. **128 步不是随意之数**——它是在给定粒子数下，全量存储式 BPTT 能在 GPU 显存中容纳的最大时间窗口，同时覆盖喷泉从发射到命中的大致物理时长。
