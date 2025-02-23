import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig) -> None:
    print(cfg)
    model = instantiate(cfg, _recursive_=False)
    model = model.cuda()
    
    # dummy_input = torch.randn(1, 4, 3, 518, 518).cuda()
    # batch = {"images": dummy_input}
    # preds = model(batch)

    pretrain_model = torch.load("/fsx-repligen/jianyuan/cvpr2025_ckpts/r518_t7_cmh_v7_0-d4w770q.pt")
    model_dict = pretrain_model["model"]


    model.load_state_dict(model_dict, strict=False)

    batch = torch.load("/fsx-repligen/jianyuan/cvpr2025_ckpts/batch.pth")
    # y_hat_raw = torch.load("/fsx-repligen/jianyuan/cvpr2025_ckpts/y_hat.pth")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):     
            y_hat = model(batch)
    
    
    # torch.save(preds, "/fsx-repligen/jianyuan/cvpr2025_ckpts/new_y_hat.pth")


    # visual_dict = {
    #     "extrinsics": y_hat["pred_extrinsic"].detach() if "pred_extrinsic" in y_hat else batch["extrinsics"],
    #     "world_points": y_hat["pred_world_points"].detach() if "pred_world_points" in y_hat else batch["world_points"],
    #     "world_points_conf": y_hat["pred_world_points_conf"].detach() if "pred_world_points_conf" in y_hat else None,
    #     "depths": y_hat["pred_depth"].detach() if "pred_depth" in y_hat else batch["depths"],
    #     "depths_conf": y_hat["pred_depth_conf"].detach() if "pred_depth_conf" in y_hat else None,
    #     "images": batch["images"],
    #     "intrinsics": batch["intrinsics"],
    #     "gt_extrinsics": batch["extrinsics"],
    #     "gt_world_points": batch["world_points"],
    #     "gt_depths": batch["depths"],
    #     "gt_intrinsics": batch["intrinsics"],
    # }

    # preds["pred_world_points"]
    # y_hat["pred_world_points"]
    # torch.save(visual_dict, f"/fsx-repligen/jianyuan/cvpr2025_ckpts/visual_99.pth")
    import pdb; pdb.set_trace()
    
    print(model)


if __name__ == "__main__":
    main()