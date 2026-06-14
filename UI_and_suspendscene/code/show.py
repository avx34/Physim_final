import numpy as np
import matplotlib.pyplot as plt

# 读取npz
log_data = np.load("data/training_log.npz")

# 取出对应数组
epoch_arr = log_data["epoch"]
loss_arr = log_data["loss"]
E_pred_arr = log_data["E_pred"]
error_E_arr = log_data["E_abs_error"]

# 截取前550轮
mask = epoch_arr <= 500
ep = epoch_arr[mask]
loss = loss_arr[mask]
E_pred = E_pred_arr[mask]
err_E = error_E_arr[mask]

E_gt = 400.0

# 画布
fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=120)

# 1. 预测弹性模量 vs 真值
axes[0].plot(ep, E_pred, color="#2171b5", linewidth=1.2, label=r"$E_{\mathrm{pred}}$")
axes[0].axhline(y=E_gt, color="#d73027", linestyle="--", linewidth=1.5, label=r"Ground truth $E^*=400$")
axes[0].set_ylabel(r"Elastic Modulus $E$")
axes[0].set_title(r"Predicted Elastic Modulus over Training Epochs (first 550)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. 绝对误差
axes[1].plot(ep, err_E, color="#238b45", linewidth=1.2)
axes[1].set_ylabel(r"$|E - E^*|$")
axes[1].set_title(r"Absolute Error of Modulus Estimation")
axes[1].grid(alpha=0.3)

# 3. 损失曲线
axes[2].plot(ep, loss, color="#756bb1", linewidth=1.2)
axes[2].set_xlabel(r"Epoch")
axes[2].set_ylabel(r"Loss Value")
axes[2].set_title(r"Training Loss Curve")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("suspend_nnfd_convergence.png", bbox_inches="tight")
plt.show()