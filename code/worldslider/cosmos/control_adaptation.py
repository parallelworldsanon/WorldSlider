"""Stage 1 - control adaptation: teach Cosmos-Transfer2.5 to read FlightGrid.

Supervised training on (FlightGrid, caption, optional first frame) -> RGB video.
The FlightGrid video enters through the pre-trained depth-control branch: the
model is initialised from the official depth-control checkpoint, and only the
control branch is optimised; the video DiT, VAE and text encoder stay frozen.

Install this file as
    cosmos_transfer2/experiments/worldslider/control_adaptation.py
inside a cosmos-transfer2.5 checkout with `flightgrid_control.patch` applied,
then launch

    torchrun --nproc_per_node=<N> -m scripts.train \
        --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
        -- experiment=worldslider_control_adaptation

Dataset layout (see scripts/prepare_dataset.py):
    <WORLDSLIDER_DATASET>/videos/<clip>.mp4       target RGB, 93 frames
    <WORLDSLIDER_DATASET>/flightgrid/<clip>.mp4   FlightGrid render of the same poses
    <WORLDSLIDER_DATASET>/captions/<clip>.json    {"caption": "..."} (environment only)
"""
import os

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader
from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset import SingleViewTransferDataset
from cosmos_transfer2.config import DEFAULT_BASE_EXPERIMENT, MODEL_CHECKPOINTS, ModelKey, ModelVariant

DATASET_DIR = os.environ.get("WORLDSLIDER_DATASET", "datasets/worldslider")
# "480p" with a square source maps to 640x640, the native resolution of our
# training windows; the resolution key must be set on model.config, since the
# base experiment binds the dataset resolution to it.
RESOLUTION = os.environ.get("WORLDSLIDER_RES", "480p")
NUM_FRAMES = 93
BATCH_SIZE = int(os.environ.get("WORLDSLIDER_BS", "1"))     # per GPU; multi-GPU is data parallel
MAX_ITER = int(os.environ.get("WORLDSLIDER_MAX_ITER", "20000"))
SAVE_ITER = int(os.environ.get("WORLDSLIDER_SAVE_ITER", "500"))
EVAL_EVERY = int(os.environ.get("WORLDSLIDER_EVAL_EVERY", "500"))

DEPTH_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(variant=ModelVariant.DEPTH)]


def _sampler(dataset):
    """Distributed sampler whose permutation advances every epoch and differs
    between runs (seeded by WORLDSLIDER_DATA_SEED), so resumed jobs do not
    replay the same head of one permutation."""
    from megatron.core import parallel_state
    from torch.utils.data.distributed import DistributedSampler

    class _EpochSampler(DistributedSampler):
        def __iter__(self):
            it = super().__iter__()
            self.epoch += 1
            return it

    return _EpochSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=int(os.environ.get("WORLDSLIDER_DATA_SEED", "0")),
    )


def register_data(name: str, hint_key: str = "control_input_flightgrid") -> str:
    ds = L(SingleViewTransferDataset)(
        dataset_dir=DATASET_DIR,
        num_frames=NUM_FRAMES,
        video_size=(640, 640),
        resolution=RESOLUTION,
        hint_key=hint_key,
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
    )
    ConfigStore.instance().store(
        group="data_train",
        package="dataloader_train",
        name=name,
        node=L(get_generic_dataloader)(
            dataset=ds,
            sampler=L(_sampler)(dataset=ds) if dist.is_initialized() else None,
            batch_size=BATCH_SIZE,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
    return name


def _callbacks(eval_every: int):
    off_cloud = dict(save_s3=False)
    sample = dict(save_s3=False, every_n=eval_every)
    return dict(
        heart_beat=off_cloud,
        iter_speed=off_cloud,
        device_monitor=off_cloud,
        dataloader_speed=off_cloud,
        frame_loss_log=off_cloud,
        wandb=off_cloud,
        wandb_10x=off_cloud,
        every_n_sample_reg=sample,
        every_n_sample_ema=sample,
    )


def make_experiment(name: str, data_group: str, max_iter: int = MAX_ITER, save_iter: int = SAVE_ITER):
    return dict(
        defaults=[
            DEFAULT_BASE_EXPERIMENT,
            {"override /data_train": data_group},
            {"override /callbacks": ["basic", "viz_online_sampling", "wandb", "cluster_speed"]},
        ],
        job=dict(project="worldslider", group="control_adaptation", name=name),
        checkpoint=dict(
            save_iter=save_iter,
            load_path=DEPTH_CHECKPOINT.s3.uri,   # resolved to the local HF download
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        model=dict(
            config=dict(
                hint_keys="flightgrid",
                base_load_from=None,
                resolution=RESOLUTION,
                # train text-to-video and image-to-video conditioning only
                min_num_conditional_frames=0,
                max_num_conditional_frames=1,
                conditional_frames_probs={0: 0.5, 1: 0.5},
            )
        ),
        trainer=dict(
            max_iter=max_iter,
            straggler_detection=dict(enabled=False),
            callbacks=_callbacks(EVAL_EVERY),
        ),
        model_parallel=dict(context_parallel_size=int(os.environ.get("WORLDSLIDER_CP", "1"))),
    )


worldslider_control_adaptation = make_experiment(
    "worldslider_control_adaptation", register_data("worldslider_data_control_adaptation")
)

ConfigStore.instance().store(
    group="experiment", package="_global_", name="worldslider_control_adaptation", node=worldslider_control_adaptation
)
