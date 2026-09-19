"""Stage 2 - DMD distillation of the control-adapted model into a 4-step generator.

Uses the in-tree DMD2 recipe of cosmos-transfer2.5 with the stage-1 model as
the frozen teacher and FlightGrid as the control condition. The teacher is a
consolidated `.pt` checkpoint (flat `net.*` keys, EMA weights, bf16), which is
the format the distillation model loader expects; convert a DCP checkpoint with
the repository's consolidation tool first.

    torchrun --standalone --nproc_per_node=<N> -m scripts.train \
        --config=cosmos_transfer2/_src/interactive/configs/registry_transfer2p5.py \
        -- experiment=worldslider_dmd

Two settings are inherited from the recipe rather than set here, and both differ from
stage 1: the number of conditional latent frames is sampled from {0: 0.6, 1: 0.2, 2: 0.2}
(stage 1 is restricted to 0 or 1), and each generator step simulates a random number of
sampling steps between 1 and 4 rather than always four. Override them in `overrides` if
you want stage-1 conditioning here.
"""
import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.interactive.configs.registry_experiment.experiments_dmd2_transfer2p5 import make_experiment
from cosmos_transfer2.experiments.worldslider.control_adaptation import RESOLUTION, register_data

TEACHER = os.environ.get("WORLDSLIDER_TEACHER", "checkpoints/worldslider_control_adaptation_ema_bf16.pt")
EVAL_EVERY = int(os.environ.get("WORLDSLIDER_EVAL_EVERY", "250"))

worldslider_dmd = make_experiment(
    name="worldslider_dmd",
    cp_size=1,
    fsdp_size=int(os.environ.get("WORLDSLIDER_FSDP", "4")),
    overrides=dict(
        job=dict(project="worldslider", group="dmd", name="worldslider_dmd"),
        upload_reproducible_setup=False,
        checkpoint=dict(
            save_iter=int(os.environ.get("WORLDSLIDER_SAVE_ITER", "500")),
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=int(os.environ.get("WORLDSLIDER_MAX_ITER", "5000")),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=EVAL_EVERY, num_samples_per_prompt=1),
                every_n_sample_ema=dict(every_n=EVAL_EVERY, num_samples_per_prompt=1),
            ),
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(load_path=TEACHER, credentials=None),
                condition_postprocessor=dict(hint_keys=["flightgrid"]),
                resolution=RESOLUTION,
            )
        ),
    ),
)

# Replace the recipe's mock web-dataset group with our local dataset. The
# defaults list is a LazyDict, so entries are DictConfig rather than dict.
_DATA = register_data("worldslider_data_dmd")
for _i, _entry in enumerate(worldslider_dmd["defaults"]):
    try:
        keys = list(_entry.keys())
    except Exception:
        continue
    if "override /data_train" in keys:
        worldslider_dmd["defaults"][_i] = {"override /data_train": _DATA}
        break
else:
    raise RuntimeError("data_train entry not found in the DMD2 defaults")
worldslider_dmd["dataloader_train"] = dict(num_workers=4)

ConfigStore.instance().store(group="experiment", package="_global_", name="worldslider_dmd", node=worldslider_dmd)
