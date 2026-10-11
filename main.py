import sys
import hydra

from omegaconf import DictConfig, OmegaConf

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from time import time

from loguru import logger

import traceback
import os
from lightning import Trainer

from gspp.predictor import PertPredictor
from gspp.data.graphmodule import GSPGraph
from gspp.data.datamodule import PertDataModule

from gspp.utils import set_seed
from gspp.constants import CHECKPOINT_DIR
from gspp.utils import download_files_from_zenodo


# Main function to initialize Hydra and parse the configuration
@hydra.main(config_name="config-gat", config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    if os.environ.get("TXPERT_SKIP_DOWNLOAD", "0") != "1":
        download_files_from_zenodo("15420279")
    try:
        t0 = time()
        infer(cfg)
        logger.info(f"Time taken: {time() - t0} seconds")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def infer(cfg: DictConfig):
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"

    set_seed(cfg.seed)

    data_args = OmegaConf.to_container(cfg.datamodule, resolve=True)

    datamodule = PertDataModule(**data_args)
    datamodule.prepare_data()

    # Instatiate graph(s) used by the GNN
    graph = GSPGraph(
        pert2id=datamodule.pert2id,
        gene2id=datamodule.gene2id,
        **cfg.graph,
    )

    trainer_args = OmegaConf.to_container(cfg.get("trainer", {}), resolve=True)
    trainer_args.setdefault("accelerator", "gpu" if gpu_available else "cpu")
    trainer_args.setdefault("devices", 1)

    if cfg.mode == "train":
        checkpoint_dir = cfg.get("checkpoint_dir", "./outputs/checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{train_loss_epoch:.4f}",
            save_last=True,
            save_top_k=-1,
        )
        trainer_args.setdefault("callbacks", []).append(checkpoint_callback)

    trainer = Trainer(**trainer_args)

    model_args = OmegaConf.to_container(cfg.model, resolve=True)

    if cfg.mode == "train":
        GSP_model = PertPredictor(
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            model_args=model_args,
            adata_output_dim=datamodule.adata_output_dim,
            graph=graph,
            device=device,
            lr=cfg.get("lr", 1e-3),
            min_lr=cfg.get("min_lr", 0.0),
            lr_scheduler_args=(
                OmegaConf.to_container(cfg.get("lr_scheduler"), resolve=True)
                if isinstance(cfg.get("lr_scheduler"), DictConfig)
                else (cfg.get("lr_scheduler") or {})
            ),
            weight_decay=cfg.get("weight_decay", 0.0),
            match_cntr_for_eval=cfg.get("match_cntr_for_eval", True),
        )
        trainer.fit(GSP_model, datamodule=datamodule)
        return

    if cfg.mode == "baseline":
        datamodule.setup("fit")
        GSP_model = PertPredictor(
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            model_args=model_args,
            adata_output_dim=datamodule.adata_output_dim,
            device=device,
            match_cntr_for_eval=cfg.get("match_cntr_for_eval", True),
        ).to(device)

        GSP_model.model.prepare_baseline(**GSP_model.model.parse_inputs(datamodule))

    elif cfg.mode == "inference":
        GSP_model = PertPredictor.load_from_checkpoint(
            CHECKPOINT_DIR / cfg.checkpoint_name,
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            adata_output_dim=datamodule.adata_output_dim,
            num_perts=len(datamodule.pert2id),
            graph=graph,
            pert_names=list(datamodule.pert2id.keys()),
            model_args=model_args,
            device=device,
        ).to(device)
        # Testing
        trainer.test(GSP_model, datamodule)
    # Predictions
    elif cfg.mode == "predict":
        GSP_model = PertPredictor.load_from_checkpoint(
            CHECKPOINT_DIR / cfg.checkpoint_name,
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            adata_output_dim=datamodule.adata_output_dim,
            num_perts=len(datamodule.pert2id),
            graph=graph,
            pert_names=list(datamodule.pert2id.keys()),
            model_args=model_args,
            device=device,
        ).to(device)
        datamodule.setup("test")
        GSP_model._predict_save_dir = cfg.save_dir
        GSP_model._predict_split_name = "test"
        trainer.predict(GSP_model, datamodule)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")




if __name__ == "__main__":
    main()
