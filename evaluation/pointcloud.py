# https://github.com/facebookresearch/vggt/issues/208
# https://github.com/doubleZ0108/GeoMVSNet
# https://github.com/yyfz/Pi3/blob/evaluation/mv_recon/eval.py

import os
import json
import torch
import numpy as np
import open3d as o3d
import os.path as osp
import hydra

from omegaconf import DictConfig
from evaluation.utils.interfaces import infer_mv_pointclouds
from evaluation.utils.mv_recon import umeyama, accuracy, completion
from evaluation.utils.messages import set_default_arg, write_csv
from evaluation.utils.vis_utils import save_image_grid_auto
from evaluation.utils import load_model


@hydra.main(version_base="1.2", config_path="./configs", config_name="eval")
def main(hydra_cfg: DictConfig):

    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/relpose-angular.yaml

    # 0. create model
    # model = VGGT.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()
    model = load_model(
        device=hydra_cfg.device,
        # model_path="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/pretrained_models/model_tracker_fixed_e20.pt",
        # big_model=True,
        # model_path="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/training/logs/mum_exp004/ckpts/checkpoint.pt",
        # model_path="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/training/logs/dinov3_exp004/ckpts/checkpoint.pt",
        model_path="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/training/logs/crocov2_exp001/ckpts/checkpoint.pt",
        big_model=False,
        # encoder="dinov3"
        # encoder="mum"
        encoder="crocov2"
    )
    print(f"Loaded VGGT from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.2 ready for output directory & metrics
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        all_data_dict = {
            "Acc-mean":  0.0,  "Acc-med":  0.0,
            "Comp-mean": 0.0,  "Comp-med": 0.0,
            "NC-mean":   0.0,  "NC-med":   0.0,
            "NC1-mean":  0.0,  "NC1-med":  0.0,
            "NC2-mean":  0.0,  "NC2-med":  0.0,
        }

        # 1.3 load pre-sampled seq-id-map
        print(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Multi-View Pointcloud Reconstruction of Pi3 on dataset {dataset_name}...")
        sample_config: DictConfig = dataset_info.sampling
        print(f"Sampling strategy: {sample_config.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)

        if osp.exists(osp.join(output_root, "_all_samples.csv")):
            os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            # 2. load data, choose specific ids of a sequence
            data = dataset.get_data(sequence_name=seq_name, ids=ids)
            filelist: list         = data['image_paths']  # [str] * N
            images: torch.Tensor   = data['images']       # (N, 3, H, W)
            gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

            # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
            data_h, data_w         = images.shape[-2:]
            pred_pts: np.ndarray   = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            # 4. save input images
            seq_name = seq_name.replace("/", "-")
            save_image_grid_auto(images, osp.join(output_root, f"{seq_name}.png"))
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            pred_pts = np.ascontiguousarray(pred_pts, dtype=np.float64)
            gt_pts = np.ascontiguousarray(gt_pts, dtype=np.float64)
            colors = np.ascontiguousarray(colors, dtype=np.float64)

            # 7. save predicted & ground truth point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

            # 8. ICP align refinement
            if "DTU" in dataset_name:
                threshold = 100
            else:
                threshold = 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)
            
            # 9. estimate normals
            pcd.estimate_normals()
            pcd_gt.estimate_normals()
            pred_normal = np.asarray(pcd.normals)
            gt_normal = np.asarray(pcd_gt.normals)

            # o3d.io.write_point_cloud(
            #     os.path.join(
            #         save_path, f"{seq.replace('/', '_')}-mask-icp.ply"
            #     ),
            #     pcd,
            # )

            # 10. compute metrics
            acc, acc_med, nc1, nc1_med = accuracy(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            comp, comp_med, nc2, nc2_med = completion(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            print(
                f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            )

            # 11. save metrics to csv
            write_csv(osp.join(output_root, f"_all_samples.csv"), {
                "seq":       seq_name,
                "Acc-mean":  acc,
                "Acc-med":   acc_med,
                "Comp-mean": comp,
                "Comp-med":  comp_med,
                "NC1-mean":  nc1,
                "NC1-med":   nc1_med,
                "NC2-mean":  nc2,
                "NC2-med":   nc2_med,
            })
            all_data_dict["Acc-mean"]  += acc
            all_data_dict["Acc-med"]   += acc_med
            all_data_dict["Comp-mean"] += comp
            all_data_dict["Comp-med"]  += comp_med
            all_data_dict["NC-mean"]   += (nc1 + nc2) / 2
            all_data_dict["NC-med"]    += (nc1_med + nc2_med) / 2
            all_data_dict["NC1-mean"]  += nc1
            all_data_dict["NC1-med"]   += nc1_med
            all_data_dict["NC2-mean"]  += nc2
            all_data_dict["NC2-med"]   += nc2_med

            # release cuda memory
            torch.cuda.empty_cache()

        num_samples = len(dataset)
        metric_dict = {
            metric: value / num_samples
            for metric, value in all_data_dict.items()
            if metric != "model"
        }

        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        print(metric_dict)
        write_csv(statistics_file, metric_dict)
    
    del model
    torch.cuda.empty_cache()
    print(f"Finished evaluating Pi3 on all datasets.")


if __name__ == "__main__":
    # set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()