import numpy as np

def decompose_extrinsic(extrinsic):
    E = np.asarray(extrinsic)
    if E.shape == (4, 4):
        R, t = E[:3, :3], E[:3, 3]
    elif E.shape == (3, 4):
        R, t = E[:, :3], E[:, 3]
    else:
        raise ValueError("Extrinsic must be 3x4 or 4x4")
    C_w = -R.T @ t
    return R, t, C_w

def project_world_point(K, R, t, Xw, im_w, im_h):
    Xc = R @ Xw + t
    Z = Xc[2]
    if Z <= 1e-6:
        return None, None, False
    uvw = K @ Xc
    u, v = uvw[0]/Z, uvw[1]/Z
    in_bounds = (0 <= u < im_w) and (0 <= v < im_h)
    return (float(u), float(v)), float(Z), in_bounds

def backproject_pixel_to_world(K, R, t, depth_map, uv):
    u, v = uv
    h, w = depth_map.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    d = float(depth_map[vi, ui])
    if d <= 0:
        return None
    Kinv = np.linalg.inv(K)
    x_cam = d * (Kinv @ np.array([u, v, 1.0], dtype=np.float64))
    Xw = R.T @ (x_cam - t)
    return Xw

def beams_for_click(source_cam, uv_src, cameras, depth_maps, depth_tol_ratio=0.05):
    """
    Input:
      source_cam: {"id","K","extrinsic","image_size"}
      uv_src: (u,v) in source_cam's ORIGINAL pixel coords
      cameras: same schema list
      depth_maps: dict cam_id -> depth[h,w] (Z-depth in camera coords)
    Returns:
      Xw (3,), overlays: dict cam_id -> {"p0": (u,v), "p1": (u,v)} in ORIGINAL pixel coords
                         p0 = projected source camera center, p1 = projected world point
    """
    from .laser_mark import decompose_extrinsic, project_world_point  # self-import safe if packaged flat

    K_s = source_cam["K"]
    R_s, t_s, Cw_s = decompose_extrinsic(source_cam["extrinsic"])
    dm_s = depth_maps[source_cam["id"]]
    Xw = backproject_pixel_to_world(K_s, R_s, t_s, dm_s, uv_src)
    if Xw is None:
        return None, {}

    overlays = {}
    for cam in cameras:
        cam_id = cam["id"]
        K = cam["K"]
        R, t, _ = decompose_extrinsic(cam["extrinsic"])
        w, h = cam["image_size"]

        uv_pt, Zpt, ok_pt = project_world_point(K, R, t, Xw, w, h)
        if not ok_pt:
            continue

        uv_srcCam, Zsc, ok_sc = project_world_point(K, R, t, Cw_s, w, h)
        if not ok_sc:
            continue

        dm = depth_maps.get(cam_id)
        if dm is not None:
            ui, vi = int(round(uv_pt[0])), int(round(uv_pt[1]))
            if not (0 <= ui < w and 0 <= vi < h):
                continue
            d_here = float(dm[vi, ui])
            if d_here <= 0:
                continue
            tol = max(depth_tol_ratio * Zpt, 0.01)
            if abs(Zpt - d_here) > tol:
                continue

        overlays[cam_id] = {"p0": uv_srcCam, "p1": uv_pt}

    return Xw, overlays
