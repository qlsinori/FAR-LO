import torch

import matplotlib.pyplot as plt

cmap = plt.cm.viridis

vmin = -100  # 设置颜色映射的最小值
vmax = 0

def window_partition(x, window_size_h,window_size_w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size_h, window_size_w, C)
    return windows


window_size = 7
window_size_h = 4
window_size_w = 8

shift_size = 3
shift_size_h = 0
shift_size_w = 4

H, W = 8, 16
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size_h),
            slice(-window_size_h, -shift_size_h),
            slice(-shift_size_h, None))
w_slices = (slice(0, -window_size_w),
            slice(-window_size_w, -shift_size_w),
            slice(-shift_size_w, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

mask_windows = window_partition(img_mask, window_size_h,window_size_w)  # nW, window_size, window_size, 1
mask_windows = mask_windows.view(-1, window_size_h * window_size_w)

attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

plt.matshow(img_mask[0, :, :, 0].numpy())
plt.matshow(attn_mask[0].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.matshow(attn_mask[1].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.matshow(attn_mask[2].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.matshow(attn_mask[3].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar()

plt.show()
