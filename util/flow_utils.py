import torch
import torch.nn.functional as F

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
            
    Returns:
        Tensor: warped image or feature map
    """
    if flow.dim() == 4 and flow.shape[1] == 2:
        # print("Permuting flow from (N, 2, H, W) to (N, H, W, 2)")
        flow = flow.permute(0, 2, 3, 1)  # [N, 2, H, W] -> [N, H, W, 2]

    assert x.size()[-2:] == flow.size()[1:3]
    _, _, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=False)
    return output

def warp_error(of_model, current_frame, prev_frame, current_gt, prev_gt, use_occlusion_mask=True):
    flow_forward, flow_backward = get_flow_forward_backward(of_model, current_gt, prev_gt)
    prev_warped = flow_warp(prev_frame, flow_forward)
    prev_gt_warped = flow_warp(prev_gt, flow_forward)
    if use_occlusion_mask:
        mask = detect_occlusion(flow_forward, flow_backward)
        valid_pixels = torch.sum(mask == 1)
        mean_error = torch.sum((mask*current_frame - mask * prev_warped)**2) / (valid_pixels*3+1e-10)
    else:
        mean_error = ((current_frame - prev_warped)**2).mean()
    return mean_error

def get_flow(of_model, target, source, rescale_factor=1):
    flows = of_model(target, source)
    flow = flows[-1]
    flow = F.interpolate(flow//rescale_factor, scale_factor=1/rescale_factor, mode='bilinear') if rescale_factor != 1 else flow
    flow = flow.permute(0, 2, 3, 1) # permute to B, H, W, 2
    return flow

def compute_flow_magnitude(flow):
    flow_mag = flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2
    return flow_mag

def compute_flow_gradients(flow):

    B = flow.shape[0]
    H = flow.shape[1]
    W = flow.shape[2]

    flow_x_du = torch.zeros((B, H, W)).to('cuda')
    flow_x_dv = torch.zeros((B, H, W)).to('cuda')
    flow_y_du = torch.zeros((B, H, W)).to('cuda')
    flow_y_dv = torch.zeros((B, H, W)).to('cuda')
    
    flow_x = flow[:, :, :, 0]
    flow_y = flow[:, :, :, 1]

    flow_x_du[:, :, :-1] = flow_x[:, :, :-1] - flow_x[:, :, 1:]
    flow_x_dv[:, :-1, :] = flow_x[:, :-1, :] - flow_x[:, 1:, :]
    flow_y_du[:, :, :-1] = flow_y[:, :, :-1] - flow_y[:, :, 1:]
    flow_y_dv[:, :-1, :] = flow_y[:, :-1, :] - flow_y[:, 1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

def detect_occlusion(fw_flow, bw_flow):
    # inputs: flow_forward, flow_backward
    # return: occlusion mask 
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    tmp = bw_flow # to this for divergence between their and my interpretation of forward and backward of
    bw_flow = fw_flow
    fw_flow = tmp
    
    fw_flow_w = flow_warp(fw_flow.permute(0,3,1,2), bw_flow).permute(0,2,3,1)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5
    
    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = torch.logical_or(mask1, mask2)
    occlusion = torch.ones((fw_flow.shape[0], fw_flow.shape[1], fw_flow.shape[2])).to('cuda')
    occlusion[mask == 1] = 0

    return occlusion

import matplotlib.pyplot as plt

def save_single_occlusion_mask(occlusion_tensor, save_path='occlusion_mask.png'):
    """
    Save the first occlusion mask in the batch as a grayscale PNG image at its true resolution.
    """
    mask = occlusion_tensor[0].detach().cpu().numpy()  # shape: [H, W]
    H, W = mask.shape

    # 設定 figure 大小為實際像素大小（以 DPI 來計算）
    dpi = 100
    figsize = (W / dpi, H / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 填滿整個畫布，不留邊
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved occlusion mask to: {save_path}, size = {W}x{H}")

def get_flow_forward_backward(net, current, prev, rescale_factor=1):
    flow_forward = get_flow(net, current, prev, rescale_factor=rescale_factor)
    flow_backward = get_flow(net, prev, current, rescale_factor=rescale_factor)
    return flow_forward, flow_backward