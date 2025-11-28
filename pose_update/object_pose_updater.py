
from __future__ import annotations
import numpy as np
from typing import List, Any, Optional, Dict

import open3d as o3d

from utils.utils import _mask_to_world_pts_colors, find_object_by_id

# ------------------ 基础工具 ------------------ #
def _is_SE3(T: np.ndarray) -> bool:
    return (
        isinstance(T, np.ndarray)
        and T.shape == (4, 4)
        and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
    )


def _invert(T: np.ndarray) -> np.ndarray:
    """SE(3) 逆: [R t; 0 1]^{-1} = [R^T -R^T t; 0 1]"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """SVD 纠正旋转矩阵数值漂移"""
    U, _, Vt = np.linalg.svd(R)
    R_new = U @ Vt
    if np.linalg.det(R_new) < 0:
        U[:, -1] *= -1
        R_new = U @ Vt
    return R_new

def _make_pcd(pts: np.ndarray, colors: np.ndarray | None = None):
    """(N,3)[,(N,3)] -> o3d PointCloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        c = colors.copy()
        if c.max() > 1.5:          # 0-255 -> 0-1
            c /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return pcd


# ------------------ 核心更新函数 ------------------ #
def update_obj_pose_ee(
    objects: List[Any],
    obj_id_in_ee: int,
    T_cw: np.ndarray,               # camera → world
    T_ec: np.ndarray,               # ee → camera
) -> bool:
    if objects is None or not (0 <= obj_id_in_ee < len(objects)):
        return False

    obj = find_object_by_id(obj_id_in_ee, objects)

    for name in ("pose_init", "pose_cur"):
        if not hasattr(obj, name):
            raise AttributeError(f"Object missing required attribute '{name}'")


    # 基本合法性
    matrices = [("T_cw", T_cw), ("T_ec", T_ec), ("obj.pose_init", obj.pose_init), ("obj.pose_cur", obj.pose_cur)]
    for n, M in matrices:
        if not _is_SE3(M):
            raise ValueError(f"{n} is not a valid 4x4 SE(3) matrix")

    # 计算 ee → world
    T_ew = T_cw @ T_ec

    # 当前世界下物体位姿 object_current → world
    T_ow_current = obj.pose_cur

    # 初始化 ee → object (抓取瞬间锁定)
    if obj.T_oe is None:
        # T_eo = (ee→world)^{-1} @ (object_current→world)
        # 使得: T_ew @ T_eo = T_ow_current
        obj.T_oe = _invert(T_ew) @ T_ow_current
        if not _is_SE3(obj.T_oe):
            raise ValueError("Initialized T_eo is invalid")

    obj.pose_cur = T_ew @ obj.T_oe
    _update_related_objects_using_relative_poses(
        obj, objects
    )
    return True


# def gather_used_source_points(pcd_source: o3d.geometry.PointCloud,
#                               pcd_target: o3d.geometry.PointCloud,
#                               return_indices: bool = False,
#                               *,
#                               # 细长圆柱: 小半径、大高度
#                               slender_radius: float = 0.005,
#                               slender_height: float = 0.10,
#                               # 矮胖圆柱: 大半径、小高度
#                               squat_radius: float = 0.03,
#                               squat_height: float = 0.005,
#                               # 圆柱轴向（默认 z 轴）。可传 (3,) 向量，会自动归一化
#                               axis=(0.0, 0.0, 1.0)):
#     """
#     以每个 source 点为中心，沿给定轴向放置两个圆柱（细长/矮胖）。
#     若存在任意 target 点落入这两个圆柱中的任意一个，则“命中”该 source 点。
#     返回所有被命中的 source 子集点云（及可选的索引）。

#     参数
#     ----
#     slender_radius, slender_height : 细长圆柱的半径和高度
#     squat_radius,   squat_height   : 矮胖圆柱的半径和高度
#     axis : 圆柱轴向（世界系），长度不要求为1，函数内会归一化

#     说明
#     ----
#     为避免 O(N*M) 全量广播，先用“定半径邻域检索”生成稀疏对 (i_src, j_tgt)，
#     再在这些候选对上做向量化的轴向/径向判定。
#     """
#     src_pts = np.asarray(pcd_source.points, dtype=np.float32)
#     tgt_pts = np.asarray(pcd_target.points, dtype=np.float32)

#     if src_pts.size == 0 or tgt_pts.size == 0:
#         empty = o3d.geometry.PointCloud()
#         if return_indices:
#             return empty, np.empty((0,), dtype=np.int64)
#         return empty

#     # 归一化轴向
#     axis = np.asarray(axis, dtype=np.float32).reshape(3)
#     norm = np.linalg.norm(axis)
#     axis_u = axis / (norm + 1e-12)

#     # 邻域搜索的 3D 距离上界：保证覆盖两个圆柱的对角线距离
#     r_max = float(max(slender_radius, squat_radius))
#     h_max = float(max(slender_height, squat_height))
#     R_bound = float(np.sqrt(r_max**2 + (0.5 * h_max)**2))  # 欧氏距离上界

#     # ========= 优先使用 Open3D(core) 的定半径批量检索 =========
#     used_idx = None
#     try:
#         dev = o3d.core.Device("CPU:0")
#         tgt_t = o3d.core.Tensor(tgt_pts, device=dev)
#         src_t = o3d.core.Tensor(src_pts, device=dev)

#         nns = o3d.core.nns.NearestNeighborSearch(tgt_t)
#         # 兼容不同 Open3D 版本的 API 形态
#         try:
#             nns.fixed_radius_index(R_bound)
#             nbr_idx, nbr_dist, row_splits = nns.fixed_radius_search(src_t)
#         except TypeError:
#             nbr_idx, nbr_dist, row_splits = nns.fixed_radius_search(src_t, R_bound)

#         nbr_idx = nbr_idx.numpy().astype(np.int64)               # (P,)
#         row_splits = row_splits.numpy().astype(np.int64)         # (N_src+1,)
#         counts = np.diff(row_splits)                             # 每个 source 的邻居数量
#         src_ids = np.repeat(np.arange(src_pts.shape[0]), counts) # (P,)
#         tgt_ids = nbr_idx                                        # (P,)

#         # 计算每对 (src_i, tgt_j) 的轴向/径向分量（向量化）
#         diff = tgt_pts[tgt_ids] - src_pts[src_ids]               # (P,3)
#         t_signed = diff @ axis_u                                 # 轴向分量（带符号）
#         z_abs = np.abs(t_signed)                                 # 轴向绝对值
#         radial_vec = diff - np.outer(t_signed, axis_u)           # 去轴向后的径向向量
#         r_xy = np.linalg.norm(radial_vec, axis=1)                # 径向半径

#         # 两个圆柱的命中条件
#         hit_slender = (r_xy <= slender_radius) & (z_abs <= 0.5 * slender_height)
#         hit_squat   = (r_xy <= squat_radius)   & (z_abs <= 0.5 * squat_height)
#         hit_any     = hit_slender | hit_squat

#         used_idx = np.unique(src_ids[hit_any])

#     except Exception:
#         # ========= 回退到 SciPy cKDTree（同样无显式 for 循环） =========
#         try:
#             from scipy.spatial import cKDTree
#             tree_src = cKDTree(src_pts)
#             tree_tgt = cKDTree(tgt_pts)
#             # 稀疏距离矩阵：仅输出距离 <= R_bound 的配对
#             # output_type='coo_matrix' 需要较新的 SciPy；若旧版会报错，则自动退成 CSR 再转 COO
#             try:
#                 D = tree_src.sparse_distance_matrix(tree_tgt, max_distance=R_bound,
#                                                     output_type='coo_matrix')
#             except TypeError:
#                 D = tree_src.sparse_distance_matrix(tree_tgt, max_distance=R_bound)
#                 D = D.tocoo()

#             src_ids = D.row.astype(np.int64)
#             tgt_ids = D.col.astype(np.int64)

#             if src_ids.size == 0:
#                 used_idx = np.empty((0,), dtype=np.int64)
#             else:
#                 diff = tgt_pts[tgt_ids] - src_pts[src_ids]
#                 t_signed = diff @ axis_u
#                 z_abs = np.abs(t_signed)
#                 radial_vec = diff - np.outer(t_signed, axis_u)
#                 r_xy = np.linalg.norm(radial_vec, axis=1)

#                 hit_slender = (r_xy <= slender_radius) & (z_abs <= 0.5 * slender_height)
#                 hit_squat   = (r_xy <= squat_radius)   & (z_abs <= 0.5 * squat_height)
#                 hit_any     = hit_slender | hit_squat

#                 used_idx = np.unique(src_ids[hit_any])

#         except Exception as e:
#             raise RuntimeError(
#                 f"邻域检索失败（Open3D 与 SciPy 都不可用？）：{repr(e)}"
#             )

#     # 根据 used_idx 构造子集点云
#     pcd_subset = o3d.geometry.PointCloud()
#     pcd_subset.points = o3d.utility.Vector3dVector(src_pts[used_idx])

#     if pcd_source.has_colors():
#         cols = np.asarray(pcd_source.colors)
#         if len(cols) == len(src_pts):
#             pcd_subset.colors = o3d.utility.Vector3dVector(cols[used_idx])

#     if pcd_source.has_normals():
#         nrm = np.asarray(pcd_source.normals)
#         if len(nrm) == len(src_pts):
#             pcd_subset.normals = o3d.utility.Vector3dVector(nrm[used_idx])

#     if return_indices:
#         return pcd_subset, used_idx
#     return pcd_subset

def gather_used_source_points(pcd_source: o3d.geometry.PointCloud,
                              pcd_target: o3d.geometry.PointCloud,
                              return_indices: bool = False,
                              k: int = 5):
    """
    从 pcd_source 中选出：作为 pcd_target 中每个点的 k-NN 而被“使用过”的所有 source 点（去重）。
    使用 Open3D 的 batched KNN；无显式 for 循环。
    """
    src_pts = np.asarray(pcd_source.points)
    tgt_pts = np.asarray(pcd_target.points)

    if src_pts.size == 0 or tgt_pts.size == 0:
        empty = o3d.geometry.PointCloud()
        if return_indices:
            return empty, np.empty((0,), dtype=np.int64)
        return empty

    # k 不能超过源点数
    k = int(max(1, min(k, len(src_pts))))

    dev = o3d.core.Device("CPU:0")
    src_t = o3d.core.Tensor(src_pts.astype(np.float32), device=dev)
    tgt_t = o3d.core.Tensor(tgt_pts.astype(np.float32), device=dev)

    nns = o3d.core.nns.NearestNeighborSearch(src_t)
    _ = nns.knn_index()
    knn_inds, _ = nns.knn_search(tgt_t, k)   # knn_inds: [M, k]

    used_idx = np.unique(knn_inds.numpy().reshape(-1))

    pcd_subset = o3d.geometry.PointCloud()
    pcd_subset.points = o3d.utility.Vector3dVector(src_pts[used_idx])

    if pcd_source.has_colors():
        src_cols = np.asarray(pcd_source.colors)
        if len(src_cols) == len(src_pts):
            pcd_subset.colors = o3d.utility.Vector3dVector(src_cols[used_idx])

    if pcd_source.has_normals():
        src_nrm = np.asarray(pcd_source.normals)
        if len(src_nrm) == len(src_pts):
            pcd_subset.normals = o3d.utility.Vector3dVector(src_nrm[used_idx])

    if return_indices:
        return pcd_subset, used_idx
    return pcd_subset

def icp_translation_only(pcd_source: o3d.geometry.PointCloud,
                         pcd_target: o3d.geometry.PointCloud,
                         *,
                         max_corr_dist: float = 0.05,
                         max_iter: int = 20,
                         method: str = "point_to_plane",  # "point_to_point" | "point_to_plane"
                         trim_ratio: float = 0.0          # 0~0.5，剔除最差对应比例
                         ):
    """
    仅估计平移向量 t 的 ICP。返回 T_delta(4x4)

    要求:
      - point_to_plane 需 pcd_target 有法向量；没有则先 estimate_normals()。
    """
    assert method in ("point_to_point", "point_to_plane")
    src_pts0 = np.asarray(pcd_source.points, dtype=np.float32)
    tgt_pts  = np.asarray(pcd_target.points, dtype=np.float32)

    if src_pts0.size == 0 or tgt_pts.size == 0:
        raise ValueError("empty point cloud")

    # KDTree on target
    kdt = o3d.geometry.KDTreeFlann(pcd_target)

    # translation only
    t = np.zeros(3, dtype=np.float64)

    # prepare normals if needed
    if method == "point_to_plane":
        if not pcd_target.has_normals():
            raise ValueError("point_to_plane 需要目标点云具有法向量（请先 estimate_normals）")
        tgt_nrm = np.asarray(pcd_target.normals, dtype=np.float32)

    src_pts = src_pts0.copy().astype(np.float64)

    for _ in range(max_iter):
        # 1) apply current translation
        src_cur = src_pts + t   # (N,3)

        # 2) radius-NN (keep nearest within radius)
        idx_src = []
        idx_tgt = []
        for i in range(src_cur.shape[0]):
            _, idx, dist2 = kdt.search_radius_vector_3d(
                o3d.utility.Vector3dVector([src_cur[i]])[0],
                max_corr_dist
            )
            if len(idx) == 0:
                continue
            j = idx[int(np.argmin(dist2))]
            idx_src.append(i)
            idx_tgt.append(j)

        if len(idx_src) < 4:
            break

        P = src_cur[np.asarray(idx_src)]
        Q = tgt_pts[np.asarray(idx_tgt)]

        # 3) optional trimming
        if trim_ratio > 0:
            resid = np.linalg.norm(P - Q, axis=1)
            k_keep = max(3, int((1.0 - trim_ratio) * resid.size))
            keep = np.argpartition(resid, k_keep-1)[:k_keep]
            P = P[keep]; Q = Q[keep]
            if method == "point_to_plane":
                N = tgt_nrm[np.asarray(idx_tgt)][keep]

        # 4) update translation only
        if method == "point_to_point":
            t_delta = (Q.mean(axis=0) - P.mean(axis=0)).astype(np.float64)
        else:
            N = tgt_nrm[np.asarray(idx_tgt)] if trim_ratio == 0 else N  # (M,3)
            dn = (Q - P)                                                # (M,3)
            A = (N[:, :, None] * N[:, None, :]).sum(axis=0)            # (3,3)
            proj = N * np.sum(N * dn, axis=1, keepdims=True)           # (M,3)
            b = proj.sum(axis=0)                                       # (3,)
            A += 1e-9 * np.eye(3)
            t_delta = np.linalg.solve(A, b).astype(np.float64)

        new_t = t + t_delta
        if np.linalg.norm(t_delta) < 1e-6:
            t = new_t
            break
        t = new_t

    # === 输出与 Open3D 一致风格：4x4 transformation （仅平移）===
    T_delta = np.eye(4, dtype=np.float64)
    T_delta[:3, 3] = t
    return T_delta

def icp_reappear(
    obj,
    T_cw: np.ndarray,
    # T_ec: np.ndarray,
    K: np.ndarray,
    tgt_mask,
    rgb: np.ndarray = None,
    depth: np.ndarray = None,
) -> bool:
    """
    只做colored ICP，不做抓取约束。source为obj.fixed_pts/cls，target为mask提取点云。
    ICP后将变换应用到obj.pose_cur。
    """
    # target点云
    # from utils import _mask_to_world_pts_colors
    tgt_pts, tgt_cls = _mask_to_world_pts_colors(
        tgt_mask, depth, rgb, K, T_cw, sample_step=2
    )
    if len(tgt_pts) < 30:
        print("[update_obj_pose_icp] target点云太少，跳过ICP")
        return True
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(obj.points_vp.astype(np.float32))
    pcd_source.colors = o3d.utility.Vector3dVector(obj.colors_vp.astype(np.float32))
    pcd_source.transform(_invert(obj.pose_init))
    # pcd_source.points = o3d.utility.Vector3dVector(obj.latest_observation_pts.astype(np.float32))
    # pcd_source.colors = o3d.utility.Vector3dVector(obj.latest_observation_cls.astype(np.float32))
    # pcd_source.transform(_invert(obj.latest_observation_pose))
    # pcd_source.transform(obj.pose_cur)
    pcd_source.voxel_down_sample(0.002)
    pcd_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float32))
    pcd_target.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float32))
    pcd_target.transform(_invert(obj.pose_cur))
    pcd_target.voxel_down_sample(0.002)
    pcd_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )


    pcd_source = gather_used_source_points(pcd_source, pcd_target)
    # # ICP前可视化（仅显示时赋色）
    # vis_source = o3d.geometry.PointCloud(pcd_source)
    # vis_source.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source.points),1)))
    # vis_target = o3d.geometry.PointCloud(pcd_target)
    # vis_target.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target.points),1)))
    
    # # 添加世界坐标系
    # world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # o3d.visualization.draw_geometries([vis_source, vis_target, world_frame], window_name="ICP前: 绿色source, 红色target, 灰色世界坐标系")
    try:
        # reg = o3d.pipelines.registration.registration_colored_icp(
        #     pcd_source,
        #     pcd_target,
        #     max_correspondence_distance=0.05,  # 先大后小
        #     init=np.eye(4, dtype=np.float32),
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        #         relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        #     ),
        # )

        # 鲁棒 ICP
        loss = o3d.pipelines.registration.TukeyLoss(k=0.05)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        reg = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target,
            max_correspondence_distance=0.05,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            ),
        )

        # estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

        # reg = o3d.pipelines.registration.registration_generalized_icp(
        #     pcd_source, pcd_target,
        #     max_correspondence_distance=0.02,
        #     init=np.eye(4),
        #     estimation_method=estimation,
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        # )

        # T_delta = icp_translation_only(pcd_source, pcd_target)

        T_delta = reg.transformation  # source → target
        print("[update_obj_pose_icp] ICP T_delta=\n", T_delta)
        print("registrated reappeared object")
        # print(f"[ICP] fitness={reg.fitness:.8f}, rmse={reg.inlier_rmse:.8f}")

        translation_distance = np.linalg.norm(T_delta[:3, 3])
        
        # 旋转角度（从旋转矩阵提取）
        R = T_delta[:3, :3]
        # 使用旋转矩阵的迹计算角度：tr(R) = 1 + 2*cos(theta)
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值稳定性
        rotation_angle = np.arccos(cos_theta)
        if translation_distance > 0.05 or rotation_angle > 0.2:
            print("icp pose change too large, rotation angle: ", {rotation_angle}, "distance:", {translation_distance})
            return False
        if reg.fitness < 0.9:
            
            print("Warning: ICP未收敛，fitness过低！")
            return False
        if reg.inlier_rmse < 0.01:
            print("ICP performed")
            obj.pose_cur = obj.pose_cur @ T_delta
            # obj.T_oe = _invert(T_cw @ T_ec) @ obj.pose_cur
            obj.pose_cur[:3, :3] = _orthonormalize(obj.pose_cur[:3, :3])
            # ICP后可视化
            # # 重新变换source点云
            # pcd_source_after = o3d.geometry.PointCloud()
            # pcd_source_after.points = o3d.utility.Vector3dVector(obj.points_vp.astype(np.float32))
            # pcd_source_after.colors = o3d.utility.Vector3dVector(obj.colors_vp.astype(np.float32))
            # pcd_source_after.transform(_invert(obj.pose_init))
            # # pcd_source_after.points = o3d.utility.Vector3dVector(obj.latest_observation_pts.astype(np.float32))
            # # pcd_source_after.colors = o3d.utility.Vector3dVector(obj.latest_observation_cls.astype(np.float32))
            # # pcd_source_after.transform(_invert(obj.latest_observation_pose))
            # pcd_source_after.transform(obj.pose_cur)
            # pcd_source_after.voxel_down_sample(0.01)
            # pcd_source_after.estimate_normals(
            #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            # )
            # vis_source_after = o3d.geometry.PointCloud(pcd_source_after)
            # vis_source_after.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source_after.points),1)))
            # vis_target_after = o3d.geometry.PointCloud(pcd_target)
            # vis_target_after.transform(obj.pose_cur @ _invert(T_delta))
            # vis_target_after.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target_after.points),1)))
            
            # # # 添加世界坐标系
            # # world_frame_after = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            
            # o3d.visualization.draw_geometries([vis_source_after, vis_target_after], window_name="ICP后: 绿色source, 红色target, 灰色世界坐标系")
            return True
        else:
            print("ICP rmse too large, skip")
            return False
            # obj.to_be_rebuild = True
            # 平移距离

    except Exception as e:
        print(f"[update_obj_pose_icp] ICP refinement skipped: {e}")
    return True




def update_obj_pose_icp(
    objects: list,
    obj_id_in_ee: int,
    # relations: Dict[int, Dict[str, List[int]]],
    T_cw: np.ndarray,
    T_ec: np.ndarray,
    K: np.ndarray,
    masks: list = None,
    rgb: np.ndarray = None,
    depth: np.ndarray = None,
) -> bool:
    """
    只做colored ICP，不做抓取约束。source为obj.fixed_pts/cls，target为mask提取点云。
    ICP后将变换应用到obj.pose_cur。
    """
    if objects is None or not (0 <= obj_id_in_ee < len(objects)):
        return False
    obj = find_object_by_id(obj_id_in_ee, objects)
    if getattr(obj, "fixed_pts", None) is None or len(obj.fixed_pts) < 30:
        print("[update_obj_pose_icp] fixed_pts为空或太少，跳过ICP")
        return True
    if masks is None or rgb is None or depth is None:
        return True
    # 找到与obj label匹配的mask
    label = getattr(obj, "label", None)
    tgt_mask = None
    for m in masks:
        if m.get("id") == obj_id_in_ee:
            tgt_mask = m["mask"]
            break
    if tgt_mask is None:
        print(f"[update_obj_pose_icp] 未找到label={label}的mask，跳过ICP")
        return True
    # target点云
    # from utils import _mask_to_world_pts_colors
    tgt_pts, tgt_cls = _mask_to_world_pts_colors(
        tgt_mask, depth, rgb, K, T_cw, sample_step=2
    )
    if len(tgt_pts) < 30:
        print("[update_obj_pose_icp] target点云太少，跳过ICP")
        return True
    # import open3d as o3d
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(obj.fixed_pts.astype(np.float32))
    pcd_source.colors = o3d.utility.Vector3dVector(obj.fixed_cls.astype(np.float32))
    pcd_source.transform(_invert(obj.fixed_pose))
    # pcd_source.transform(obj.pose_cur)
    pcd_source.voxel_down_sample(0.001)
    pcd_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float32))
    pcd_target.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float32))
    pcd_target.transform(_invert(obj.pose_cur))
    pcd_target.voxel_down_sample(0.001)
    pcd_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    # # ICP前可视化（仅显示时赋色）
    # vis_source = o3d.geometry.PointCloud(pcd_source)
    # vis_source.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source.points),1)))
    # vis_target = o3d.geometry.PointCloud(pcd_target)
    # vis_target.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target.points),1)))
    # o3d.visualization.draw_geometries([vis_source, vis_target], window_name="ICP前: 绿色source, 红色target")
    try:
        # reg = o3d.pipelines.registration.registration_colored_icp(
        #     pcd_source,
        #     pcd_target,
        #     max_correspondence_distance=0.05,  # 先大后小
        #     init=np.eye(4, dtype=np.float32),
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        #         relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        #     ),
        # )

        # 鲁棒 ICP
        loss = o3d.pipelines.registration.TukeyLoss(k=0.09)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        reg = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target,
            max_correspondence_distance=0.05,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            ),
        )

        # estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

        # reg = o3d.pipelines.registration.registration_generalized_icp(
        #     pcd_source, pcd_target,
        #     max_correspondence_distance=0.02,
        #     init=np.eye(4),
        #     estimation_method=estimation,
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        # )

        T_delta = reg.transformation  # source → target
        # print("[update_obj_pose_icp] ICP T_delta=\n", T_delta)
        print(f"[ICP] fitness={reg.fitness:.8f}, rmse={reg.inlier_rmse:.8f}")
        if reg.fitness < 0.1:
            print("Warning: ICP未收敛，fitness过低！")
        if reg.inlier_rmse < 0.01:
            print("ICP performed")
            obj.pose_cur = obj.pose_cur @ T_delta
            obj.T_oe = _invert(T_cw @ T_ec) @ obj.pose_cur
            obj.pose_cur[:3, :3] = _orthonormalize(obj.pose_cur[:3, :3])
            _update_related_objects_using_relative_poses(
                obj, objects
            )
            # # ICP后可视化
            # # 重新变换source点云
            # pcd_source_after = o3d.geometry.PointCloud()
            # pcd_source_after.points = o3d.utility.Vector3dVector(obj.fixed_pts.astype(np.float32))
            # pcd_source_after.colors = o3d.utility.Vector3dVector(obj.fixed_cls.astype(np.float32))
            # pcd_source_after.transform(_invert(obj.fixed_pose))
            # pcd_source_after.transform(obj.pose_cur)
            # pcd_source_after.voxel_down_sample(0.01)
            # pcd_source_after.estimate_normals(
            #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            # )
            # vis_source_after = o3d.geometry.PointCloud(pcd_source_after)
            # vis_source_after.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source_after.points),1)))
            # vis_target_after = o3d.geometry.PointCloud(pcd_target)
            # vis_target_after.transform(obj.pose_cur @ _invert(T_delta))
            # vis_target_after.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target_after.points),1)))
            # o3d.visualization.draw_geometries([vis_source_after, vis_target_after], window_name="ICP后: 绿色source, 红色target")
        else:
            print("ICP rmse too large, skip")
    except Exception as e:
        print(f"[update_obj_pose_icp] ICP refinement skipped: {e}")
    return True


# def update_object_pose_recursive(
#     objects: List[Any],
#     obj_id_in_ee: int,
#     relations: Dict[int, Dict[str, List[int]]],
#     T_cw: np.ndarray,
#     T_ec: np.ndarray,
#     K: np.ndarray,
#     masks: List[Dict],
#     rgb: np.ndarray,
#     depth: np.ndarray,
# ) -> None:
#     """
#     递归更新物体位姿：当更新夹爪中物体位姿时，递归更新所有相关物体的位姿
    
#     参数:
#         objects: 物体列表
#         obj_id_in_ee: 夹爪中物体的ID
#         relations: 物体关系图
#         T_cw, T_ec, K, masks, rgb, depth: 用于位姿更新的参数
#     """
#     if obj_id_in_ee is None:
#         return
    
#     # 找到夹爪中的物体
#     obj_in_ee = find_object_by_id(obj_id_in_ee, objects)
#     if obj_in_ee is None:
#         return
    
#     # 如果还没有记录相对位姿关系，则记录当前时刻的相对位姿
#     if not hasattr(obj_in_ee, 'relative_poses_recorded') or not obj_in_ee.relative_poses_recorded:
#         _record_relative_poses(obj_in_ee, objects, relations)
#         obj_in_ee.relative_poses_recorded = True
    
#     # 更新夹爪中物体的位姿
#     update_obj_pose_ee(objects, obj_id_in_ee, T_cw, T_ec, K, masks, rgb, depth)
    
#     # 使用记录下来的相对位姿关系更新所有相关物体的位姿
#     _update_related_objects_using_relative_poses(
#         obj_in_ee, objects, relations
#     )


# def _record_relative_poses(
#     obj_in_ee: Any,
#     objects: List[Any],
#     relations: Dict[int, Dict[str, List[int]]],
# ) -> None:
#     """
#     记录被操作物体与所有相关子物体的相对位姿关系
    
#     参数:
#         obj_in_ee: 被操作的物体
#         objects: 物体列表
#         relations: 物体关系图
#     """
#     if not hasattr(obj_in_ee, 'relative_poses'):
#         obj_in_ee.relative_poses = {}
    
#     # 获取所有相关的子物体ID
#     related_obj_ids = set()
#     if obj_in_ee.id in relations:
#         current_relations = relations[obj_in_ee.id]
        
#         # 包含关系：被当前物体包含的物体
#         for contained_id in current_relations.get("contain", []):
#             if contained_id < len(objects):
#                 related_obj_ids.add(contained_id)
        
#         # 支撑关系：被当前物体支撑的物体
#         for supported_id in current_relations.get("under", []):
#             if supported_id < len(objects):
#                 related_obj_ids.add(supported_id)
    
#     # 记录每个相关物体相对于被操作物体的位姿
#     for related_id in related_obj_ids:
#         related_obj = objects[related_id]
#         # 计算相对位姿：T_relative = T_obj_in_ee^(-1) @ T_related_obj
#         T_relative = np.linalg.inv(obj_in_ee.pose_cur) @ related_obj.pose_cur
#         obj_in_ee.relative_poses[related_id] = T_relative.copy()
    
#     print(f"Recorded relative poses for {len(related_obj_ids)} related objects")


def _update_related_objects_using_relative_poses(
    obj_in_ee: Any,
    objects: List[Any],
    # relations: Dict[int, Dict[str, List[int]]],
) -> None:
    """
    使用记录下来的相对位姿关系更新所有相关物体的位姿
    
    参数:
        obj_in_ee: 被操作的物体
        objects: 物体列表
        relations: 物体关系图
    """
    # if not hasattr(obj_in_ee, 'relative_poses'):
    #     return
    
    # 使用记录下来的相对位姿关系计算每个相关物体的新位姿
    for related_id, T_relative in obj_in_ee.child_objs.items():
        if related_id >= len(objects):
            continue
            
        related_obj = find_object_by_id(related_id, objects)
        # 计算新位姿：T_new = T_obj_in_ee_current @ T_relative
        related_obj.pose_cur = obj_in_ee.pose_cur @ T_relative
    
    print(f"Updated {len(obj_in_ee.child_objs)} related objects using relative poses")


# def _update_related_objects_recursive(
#     objects: List[Any],
#     current_obj_id: int,
#     relations: Dict[int, Dict[str, List[int]]],
#     T_change: np.ndarray,
#     updated_objs: set,
# ) -> None:
#     """
#     递归更新与当前物体有contain或under关系的物体位姿
    
#     参数:
#         objects: 物体列表
#         current_obj_id: 当前物体ID
#         relations: 物体关系图
#         T_change: 位姿变化矩阵
#         updated_objs: 已更新物体的集合（避免循环更新）
#     """
#     if current_obj_id not in relations:
#         return
    
#     current_relations = relations[current_obj_id]
#     # print("111111111111111111111111111111111111111111111111111")
#     # print(current_relations)
    
#     # 获取需要更新的相关物体ID
#     related_obj_ids = set()
    
#     # 包含关系：如果当前物体被包含在其他物体中，需要更新
#     for container_id in current_relations.get("under", []):
#         if container_id not in updated_objs:
#             related_obj_ids.add(container_id)
    
#     # 支撑关系：如果当前物体支撑其他物体，需要更新
#     for supported_id in current_relations.get("contain", []):
#         if supported_id not in updated_objs:
#             related_obj_ids.add(supported_id)
    
#     # 更新相关物体的位姿
#     for related_id in related_obj_ids:
#         if related_id >= len(objects):
#             continue
            
#         related_obj = objects[related_id]
        
#         # 记录更新前的位姿
#         pose_before = related_obj.pose_cur.copy()
        
#         # 应用相同的位姿变化
#         related_obj.pose_cur = T_change @ pose_before
        
#         # 标记为已更新
#         updated_objs.add(related_id)
        
#         # 递归更新该物体的相关物体
#         _update_related_objects_recursive(
#             objects, related_id, relations, T_change, updated_objs
#         )


# def update_obj_pose_ee_with_relations(
#     objects: List[Any],
#     obj_id_in_ee: int,
#     relations: Dict[int, Dict[str, List[int]]],
#     T_cw: np.ndarray,
#     T_ec: np.ndarray,
#     K: np.ndarray,
#     masks: List[Dict],
#     rgb: np.ndarray,
#     depth: np.ndarray,
# ) -> bool:
#     """
#     带关系更新的物体位姿更新函数
    
#     参数:
#         objects: 物体列表
#         obj_id_in_ee: 夹爪中物体的ID
#         relations: 物体关系图
#         T_cw, T_ec, K, masks, rgb, depth: 用于位姿更新的参数
    
#     返回:
#         是否成功更新
#     """
#     try:
#         # update_object_pose_recursive(
#         #     objects, obj_id_in_ee, relations, T_cw, T_ec, K, masks, rgb, depth
#         # )

#         return True
#     except Exception as e:
#         print(f"Error updating object pose with relations: {e}")
#         return False


def reset_relative_poses_recorded(
    objects: List[Any],
    obj_id_in_ee: int,
) -> None:
    """
    重置物体的相对位姿记录状态，用于物体被释放时
    
    参数:
        objects: 物体列表
        obj_id_in_ee: 要重置的物体ID
    """
    if obj_id_in_ee is None:
        return
        
    obj = find_object_by_id(obj_id_in_ee, objects)
    if obj is not None and hasattr(obj, 'relative_poses_recorded'):
        obj.relative_poses_recorded = False
        if hasattr(obj, 'relative_poses'):
            obj.relative_poses.clear()
        print(f"Reset relative poses recorded for object {obj_id_in_ee}")


def update_child_objects_pose_icp(
    objects: List[Any],
    obj_id_in_ee: int,
    relations: Dict[int, Dict[str, List[int]]],
    T_cw: np.ndarray,
    T_ec: np.ndarray,
    K: np.ndarray,
    masks: List[Dict],
    rgb: np.ndarray,
    depth: np.ndarray,
) -> None:
    """
    在操作obj_in_ee过程中，使用ICP更新其子物体的位姿
    
    参数:
        objects: 物体列表
        obj_id_in_ee: 被操作的物体ID
        relations: 物体关系图
        T_cw, T_ec, K, masks, rgb, depth: 用于位姿更新的参数
    """
    if obj_id_in_ee is None:
        return
    
    # 找到被操作的物体
    obj_in_ee = find_object_by_id(obj_id_in_ee, objects)
    if obj_in_ee is None:
        return
    
    # # 获取所有相关的子物体ID
    # related_obj_ids = set()
    # if obj_in_ee.id in relations:
    #     current_relations = relations[obj_in_ee.id]
        
    #     # 包含关系：被当前物体包含的物体
    #     for contained_id in current_relations.get("contain", []):
    #         if contained_id < len(objects):
    #             related_obj_ids.add(contained_id)
        
    #     # 支撑关系：被当前物体支撑的物体
    #     for supported_id in current_relations.get("under", []):
    #         if supported_id < len(objects):
    #             related_obj_ids.add(supported_id)
    
    # 为每个子物体更新位姿
    for related_id, T in obj_in_ee.child_objs.items():
        _update_single_child_pose_icp(
            objects, related_id, obj_in_ee, T_cw, T_ec, K, masks, rgb, depth
        )

def _update_single_child_pose_icp(
    objects: List[Any],
    child_id: int,
    parent_obj: Any,
    T_cw: np.ndarray,
    T_ec: np.ndarray,
    K: np.ndarray,
    masks: List[Dict],
    rgb: np.ndarray,
    depth: np.ndarray,
) -> None:
    """
    更新单个子物体的位姿，使用ICP方法
    
    参数:
        objects: 物体列表
        child_id: 子物体ID
        parent_obj: 父物体
        T_cw, T_ec, K, masks, rgb, depth: 用于位姿更新的参数
    """
    if child_id >= len(objects):
        return
        
    child_obj = find_object_by_id(child_id, objects)
    
    # 找到与子物体匹配的mask
    tgt_mask = None
    for m in masks:
        if m.get("id") == child_id:
            tgt_mask = m["mask"]
            break
    
    if tgt_mask is None:
        return
    
    # # 如果还没有记录固定观测，则记录当前mask下的点云
    # if not hasattr(child_obj, 'fixed_pts_child') or child_obj.fixed_pts_child is None:
    #     child_obj.fixed_pts_child, child_obj.fixed_cls_child = _mask_to_world_pts_colors(
    #         tgt_mask, depth, rgb, K, T_cw, sample_step=2
    #     )
    #     child_obj.fixed_pose_child = child_obj.pose_cur.copy()
    #     print(f"Recorded fixed observation for child object {child_id}")
    #     return
    
    # 提取当前mask下的点云作为target
    tgt_pts, tgt_cls = _mask_to_world_pts_colors(
        tgt_mask, depth, rgb, K, T_cw, sample_step=2
    )
    
    if len(tgt_pts) < 30:
        return
    
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(child_obj.points_vp.astype(np.float32))
    pcd_source.colors = o3d.utility.Vector3dVector(child_obj.colors_vp.astype(np.float32))
    pcd_source.transform(_invert(child_obj.pose_init))
    # pcd_source.points = o3d.utility.Vector3dVector(obj.latest_observation_pts.astype(np.float32))
    # pcd_source.colors = o3d.utility.Vector3dVector(obj.latest_observation_cls.astype(np.float32))
    # pcd_source.transform(_invert(obj.latest_observation_pose))
    # pcd_source.transform(obj.pose_cur)
    pcd_source.voxel_down_sample(0.002)
    pcd_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float32))
    pcd_target.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float32))
    pcd_target.transform(_invert(child_obj.pose_cur))
    pcd_target.voxel_down_sample(0.002)
    pcd_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )


    pcd_source = gather_used_source_points(pcd_source, pcd_target)
    
    try:
        # 执行ICP
        loss = o3d.pipelines.registration.TukeyLoss(k=0.05)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        
        reg = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target,
            max_correspondence_distance=0.05,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            ),
        )
        
        T_delta = reg.transformation  # source → target
        
        print(f"[Child ICP {child_id}] fitness={reg.fitness:.8f}, rmse={reg.inlier_rmse:.8f}")
        
        if reg.fitness < 0.1:
            print(f"Warning: Child {child_id} ICP未收敛，fitness过低！")
            return
            
        if reg.inlier_rmse < 0.01:
            # 更新子物体位姿
            child_obj.pose_cur = child_obj.pose_cur @ T_delta
            child_obj.pose_cur[:3, :3] = _orthonormalize(child_obj.pose_cur[:3, :3])
            
            # 更新子物体与父物体的相对位姿关系
            # if hasattr(parent_obj, 'relative_poses') and child_id in parent_obj.relative_poses:
                # 重新计算相对位姿：T_relative = T_parent^(-1) @ T_child
            T_relative = np.linalg.inv(parent_obj.pose_cur) @ child_obj.pose_cur
            parent_obj.child_objs[child_id] = T_relative.copy()
            print(f"Updated relative pose for child {child_id}")
            
            print(f"Child {child_id} ICP performed")
        else:
            print(f"Child {child_id} ICP rmse too large, skip")
            
    except Exception as e:
        print(f"[_update_single_child_pose_icp] Child {child_id} ICP refinement skipped: {e}")



# def _update_single_child_pose_icp(
#     objects: List[Any],
#     child_id: int,
#     parent_obj: Any,
#     T_cw: np.ndarray,
#     T_ec: np.ndarray,
#     K: np.ndarray,
#     masks: List[Dict],
#     rgb: np.ndarray,
#     depth: np.ndarray,
# ) -> None:
#     """
#     更新单个子物体的位姿，使用ICP方法
    
#     参数:
#         objects: 物体列表
#         child_id: 子物体ID
#         parent_obj: 父物体
#         T_cw, T_ec, K, masks, rgb, depth: 用于位姿更新的参数
#     """
#     if child_id >= len(objects):
#         return
        
#     child_obj = find_object_by_id(child_id, objects)
    
#     # 找到与子物体匹配的mask
#     tgt_mask = None
#     for m in masks:
#         if m.get("id") == child_id:
#             tgt_mask = m["mask"]
#             break
    
#     if tgt_mask is None:
#         return
    
#     # 如果还没有记录固定观测，则记录当前mask下的点云
#     if not hasattr(child_obj, 'fixed_pts_child') or child_obj.fixed_pts_child is None:
#         child_obj.fixed_pts_child, child_obj.fixed_cls_child = _mask_to_world_pts_colors(
#             tgt_mask, depth, rgb, K, T_cw, sample_step=2
#         )
#         child_obj.fixed_pose_child = child_obj.pose_cur.copy()
#         print(f"Recorded fixed observation for child object {child_id}")
#         return
    
#     # 提取当前mask下的点云作为target
#     tgt_pts, tgt_cls = _mask_to_world_pts_colors(
#         tgt_mask, depth, rgb, K, T_cw, sample_step=2
#     )
    
#     if len(tgt_pts) < 30:
#         return
    
#     # 准备source点云（固定观测）
#     pcd_source = o3d.geometry.PointCloud()
#     pcd_source.points = o3d.utility.Vector3dVector(child_obj.fixed_pts_child.astype(np.float32))
#     pcd_source.colors = o3d.utility.Vector3dVector(child_obj.fixed_cls_child.astype(np.float32))
#     pcd_source.transform(_invert(child_obj.fixed_pose_child))
#     pcd_source.voxel_down_sample(0.001)
#     pcd_source.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
#     )
    
#     # 准备target点云（当前观测）
#     pcd_target = o3d.geometry.PointCloud()
#     pcd_target.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float32))
#     pcd_target.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float32))
#     pcd_target.transform(_invert(child_obj.pose_cur))
#     pcd_target.voxel_down_sample(0.001)
#     pcd_target.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
#     )
    
#     try:
#         # 执行ICP
#         loss = o3d.pipelines.registration.TukeyLoss(k=0.09)
#         estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        
#         reg = o3d.pipelines.registration.registration_icp(
#             pcd_source, pcd_target,
#             max_correspondence_distance=0.05,
#             init=np.eye(4),
#             estimation_method=estimation,
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#                 relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
#             ),
#         )
        
#         T_delta = reg.transformation  # source → target
        
#         print(f"[Child ICP {child_id}] fitness={reg.fitness:.8f}, rmse={reg.inlier_rmse:.8f}")
        
#         if reg.fitness < 0.1:
#             print(f"Warning: Child {child_id} ICP未收敛，fitness过低！")
#             return
            
#         if reg.inlier_rmse < 0.01:
#             # 更新子物体位姿
#             child_obj.pose_cur = child_obj.pose_cur @ T_delta
#             child_obj.pose_cur[:3, :3] = _orthonormalize(child_obj.pose_cur[:3, :3])
            
#             # 更新子物体与父物体的相对位姿关系
#             # if hasattr(parent_obj, 'relative_poses') and child_id in parent_obj.relative_poses:
#                 # 重新计算相对位姿：T_relative = T_parent^(-1) @ T_child
#             T_relative = np.linalg.inv(parent_obj.pose_cur) @ child_obj.pose_cur
#             parent_obj.child_objs[child_id] = T_relative.copy()
#             print(f"Updated relative pose for child {child_id}")
            
#             print(f"Child {child_id} ICP performed")
#         else:
#             print(f"Child {child_id} ICP rmse too large, skip")
            
#     except Exception as e:
#         print(f"[_update_single_child_pose_icp] Child {child_id} ICP refinement skipped: {e}")


def clear_child_fixed_observations(
    objects: List[Any],
    obj_id_in_ee: int,
    # relations: Dict[int, Dict[str, List[int]]],
) -> None:
    """
    清除子物体的固定观测，用于操作结束后
    
    参数:
        objects: 物体列表
        obj_id_in_ee: 被操作的物体ID
        relations: 物体关系图
    """
    if obj_id_in_ee is None:
        return
        
    obj_in_ee = find_object_by_id(obj_id_in_ee, objects)
    if obj_in_ee is None:
        return
    
    # # 获取所有相关的子物体ID
    # related_obj_ids = set()
    # if obj_in_ee.id in relations:
    #     for obj_id, rels in relations.items():
    #         if obj_id == obj_in_ee.id:
    #             current_relations = rels
        
    #     # 包含关系：被当前物体包含的物体
    #     for contained_id in current_relations.get("contain", []):
    #         if contained_id < len(objects):
    #             related_obj_ids.add(contained_id)
        
    #     # 支撑关系：被当前物体支撑的物体
    #     for supported_id in current_relations.get("under", []):
    #         if supported_id < len(objects):
    #             related_obj_ids.add(supported_id)
    
    # 清除每个子物体的固定观测
    for related_id in obj_in_ee.child_objs.keys():
        if related_id >= len(objects):
            continue
            
        child_obj = find_object_by_id(related_id, objects)
        if hasattr(child_obj, 'fixed_pts_child'):
            child_obj.fixed_pts_child = None
        if hasattr(child_obj, 'fixed_cls_child'):
            child_obj.fixed_cls_child = None
        if hasattr(child_obj, 'fixed_pose_child'):
            child_obj.fixed_pose_child = None
    
    print(f"Cleared fixed observations for {len(obj_in_ee.child_objs.keys())} child objects")
