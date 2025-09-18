# ! /usr/bin/python
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

import glob
import json
import os
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import List, Optional
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from nemo.collections.asr.parts.utils.slu_utils import SequenceGeneratorConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging
from model import DOMA

@dataclass
class InferenceConfig:
    model_path: Optional[str] = None  
    pretrained_name: Optional[str] = None 
    audio_dir: Optional[str] = None  
    dataset_manifest: Optional[str] = None  
    asr_transcripts_filepath: Optional[str] = None 
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 8
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"
    overwrite_transcripts: bool = True
    sequence_generator: SequenceGeneratorConfig = SequenceGeneratorConfig(type="greedy")

def slurp_inference(
    model,
    path2manifest: str,
    transcripts_filepath: str,
    batch_size: int = 4,
    num_workers: int = 0,
) -> List[str]:
    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)
    hypotheses = []
    mode = model.training
    device = next(model.parameters()).device
    dither_value = model.preprocessor.featurizer.dither
    pad_to_value = model.preprocessor.featurizer.pad_to
    try:
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()
        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)
        config = {
            'manifest_filepath': path2manifest,
            'transcripts_filepath': transcripts_filepath,
            'batch_size': batch_size,
            'num_workers': num_workers,
        }
        temporary_datalayer = model._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing", ncols=80):
            input_signal, input_signal_len, _, _ = test_batch["clean"]
            asr_text = test_batch["pred_asr"]
            predictions = model.predict(
                input_signal=input_signal.to(device), 
                input_signal_length=input_signal_len.to(device),
                asr_text=asr_text
            )
            hypotheses += predictions
            del predictions
            del test_batch

    finally:
        model.train(mode=mode)
        model.preprocessor.featurizer.dither = dither_value
        model.preprocessor.featurizer.pad_to = pad_to_value
        logging.set_verbosity(logging_level)

    return hypotheses


@hydra_runner(config_name="InferenceConfig", schema=InferenceConfig)
def run_inference(cfg: InferenceConfig) -> InferenceConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)
    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    logging.info(f"Restoring model : {cfg.model_path}")
    model = DOMA.restore_from(restore_path=cfg.model_path, map_location=map_location)
    model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    model.set_trainer(trainer)
    model = model.eval()
    model.set_decoding_strategy(cfg.sequence_generator)

    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
    else:
        filepaths = []
        if os.stat(cfg.dataset_manifest).st_size == 0:
            logging.error(f"The input dataset_manifest {cfg.dataset_manifest} is empty. Exiting!")
            return None

        manifest_dir = Path(cfg.dataset_manifest).parent
        with open(cfg.dataset_manifest, 'r') as f:
            has_two_fields = []
            for line in f:
                item = json.loads(line)
                if "offset" in item and "duration" in item:
                    has_two_fields.append(True)
                else:
                    has_two_fields.append(False)
                audio_file = Path(item['audio_filepath'])
                if not audio_file.is_file() and not audio_file.is_absolute():
                    audio_file = manifest_dir / audio_file
                filepaths.append(str(audio_file.absolute()))

    logging.info(f"\nStart inference with {len(filepaths)} files...\n")

    if cfg.output_filename is None:
        if cfg.audio_dir is not None:
            cfg.output_filename = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + '.json'
        else:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')

    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    with torch.amp.autocast(model.device.type, enabled=cfg.amp):
        with torch.no_grad():
            predictions = slurp_inference(
                model=model,
                path2manifest=cfg.dataset_manifest,
                transcripts_filepath=cfg.asr_transcripts_filepath,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

    logging.info(f"Finished transcribing {len(filepaths)} files !")
    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        if cfg.audio_dir is not None:
            for idx, text in enumerate(predictions):
                item = {'audio_filepath': filepaths[idx], 'pred_text': text}
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item['pred_text'] = predictions[idx]
                    f.write(json.dumps(item) + "\n")

    logging.info("Finished writing predictions !")
    return cfg

if __name__ == '__main__':
    run_inference() 
