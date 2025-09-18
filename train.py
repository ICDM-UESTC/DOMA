# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from model import DOMA
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="./config/", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = DOMA(cfg=cfg.model, trainer=trainer)
    pretraind_model = DOMA.restore_from(
        restore_path="path_to_pretrained_icsf_model",
        map_location=torch.device("cpu")
    )
    state_dict = pretraind_model.state_dict()
    model.load_state_dict(state_dict, strict=False)
    del pretraind_model

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.ap.parameters():
        param.requires_grad = True  
    for param in model.embedding.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False

    trainer.fit(model)
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if model.prepare_test(trainer):
            trainer.test(model)

if __name__ == '__main__':
    main()
