import cv2
import numpy as np


def compute_mapping(cam_W, cam_K, coords, depth, image_dim, vis_thres=0.25, cut_bound=0):
    """
    :param cam_W: 4 x 4 format
    :param cam_K: 4 x 4 format (as in habitat)
    :param coords: N x 3 format
    :param depth: H x W format
    :param image_dim: (W, H)
    :return: mapping, N x 3 format, (H,W,mask)
    """
    world_to_camera = np.linalg.inv(cam_W).T
    intrinsic = cam_K.copy()[:3, :3]
    intrinsic[0, 2] = 1
    intrinsic[1, 2] = 1
    intrinsic[0] *= image_dim[0] * 0.5 # W
    intrinsic[1] *= image_dim[1] * 0.5 # H

    mapping = np.zeros((3, coords.shape[0]), dtype=int)
    coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
    assert coords_new.shape[0] == 4, "[!] Shape error"

    p = np.matmul(world_to_camera.T, coords_new)
    p[2] = -p[2]
    p[1] = -p[1]
    p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
    p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
    pi = np.round(p).astype(int)  # simply round the projected coordinates
    center_distance = np.sqrt((pi[0] - image_dim[0] / 2) ** 2 + (pi[1] - image_dim[1] / 2) ** 2)
    inside_mask = (
        (pi[0] >= cut_bound)
        * (pi[1] >= cut_bound)
        * (pi[0] < image_dim[0] - cut_bound)
        * (pi[1] < image_dim[1] - cut_bound)
    )
    # generate depth
    if isinstance(depth, str):
        depth = np.ones((image_dim[1], image_dim[0])) * 999999
        for i in range(p.shape[1]):
            if p[2, i] > 0.2 and inside_mask[i] and depth[pi[1, i], pi[0, i]] > p[2, i]:
                depth[pi[1, i], pi[0, i]] = p[2, i]

    if depth is not None:
        depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
        occlusion_mask = (
            np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= vis_thres * depth_cur
        )

        inside_mask[inside_mask == True] = occlusion_mask
    else:
        front_mask = p[2] > 0  # make sure the depth is in front
        inside_mask = front_mask * inside_mask
    mapping[0][inside_mask] = pi[1][inside_mask]
    mapping[1][inside_mask] = pi[0][inside_mask]
    mapping[2][inside_mask] = 1
    weight = np.exp(-center_distance / 10)

    return mapping.T, weight


def filter_pc_with_mask(pc, mapping, mask):
    # pc: N x 3 or 6
    # mapping: N x 3 (H, W, mask), mask -- 1 indidate there is valid mapping
    # mask: H x W

    # Find the indices (i, j) where mask[i, j] == 1
    indices = np.argwhere(mask == 1)

    # Convert indices to a set of tuples for faster lookup
    indices_set = set(map(tuple, indices))

    # Filter out points from the point cloud
    filtered_pc = [pc[idx] for idx, (h, w, valid) in enumerate(mapping) if valid == 1 and (h, w) in indices_set]

    return np.array(filtered_pc)
def filter_pc_color_with_mask(pc, rgbs, mapping, mask):
    indices = np.argwhere(mask == 1)
    # print(indices)
    # Convert indices to a set of tuples for faster lookup
    indices_set = set(map(tuple, indices))

    # Filter out points from the point cloud
    filtered_pc = [pc[idx] for idx, (h, w, valid) in enumerate(mapping) if valid == 1 and (h, w) in indices_set]
    filtered_rgbs = [rgbs[idx] for idx, (h, w, valid) in enumerate(mapping) if valid == 1 and (h, w) in indices_set]
    # print(f'len filtered_pc is {len(filtered_pc)}')
    # filtered_pc = [pc[idx] for idx, (h, w, valid) in enumerate(mapping) if valid == 1 and (h, w) in indices_set]
    # filtered_rgbs = [rgbs[idx] for idx, (h, w, valid) in enumerate(mapping) if valid == 1 and (h, w) in indices_set]
    return np.array(filtered_pc), filtered_rgbs

def filter_mask_with_point(mask, point, erode_kernel=0):
    x, y = point
    H, W = mask.shape
    if mask[x, y] == 0:
        # If the starting point is not part of the region, flip all to 0 and return
        return np.zeros((H, W), dtype=int)

    # Create a copy of the mask to store the result
    result_mask = np.zeros((H, W), dtype=int)

    # Use OpenCV's floodFill to find the connected region
    mask_copy = mask.astype(np.uint8)
    flood_fill_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)  # Flood fill mask must be 2 pixels larger than the input
    cv2.floodFill(mask_copy, flood_fill_mask, (y, x), 2)

    # Set the connected region to 1 in the result mask
    result_mask[mask_copy == 2] = 1

    if erode_kernel > 0:
        result_mask = cv2.erode(result_mask.astype(np.float32), np.ones((erode_kernel, erode_kernel), np.uint8), iterations=1).astype(np.uint8)

    return result_mask