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

import os
from math import ceil
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.utils.slu_utils import SequenceGenerator, SequenceGeneratorConfig, get_seq_mask
from nemo.collections.common.losses import SmoothedNLLLoss
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging, model_utils
from dataset import get_triplet_bpe_dataset
import re
import shutil
import glob
from generate import generate_ap, generate
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence

__all__ = ["DOMA"]

LLADA_DEVICE = 'your_device'
LLADA_MODEL = AutoModel.from_pretrained('your_path_to_llada', trust_remote_code=True, torch_dtype=torch.bfloat16).to(LLADA_DEVICE).eval()
LLADA_TOKENIZER = AutoTokenizer.from_pretrained('your_path_to_llada', trust_remote_code=True)
for param in LLADA_MODEL.parameters():
    param.requires_grad = False

def pad_oracle_tokens(oracle_tokens, gen_length, pad_value=126081):
    processed_tokens = []
    for x in oracle_tokens:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)
        processed_tokens.append(x)
    oracle_tokens_padded = pad_sequence(processed_tokens, batch_first=True, padding_value=pad_value)
    if oracle_tokens_padded.size(1) < gen_length:
        pad_size = gen_length - oracle_tokens_padded.size(1)
        extra_pad = torch.full(
            (oracle_tokens_padded.size(0), pad_size),
            pad_value,
            dtype=oracle_tokens_padded.dtype,
            device=oracle_tokens_padded.device,
        )
        oracle_tokens_padded = torch.cat([oracle_tokens_padded, extra_pad], dim=1)
    else:
        oracle_tokens_padded = oracle_tokens_padded[:, :gen_length]
    return oracle_tokens_padded

class AdaptivePriorModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 4096,
        num_heads: int = 8,
        ff_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.gate_proj = nn.Linear(embed_dim, 1)

    def forward(self, response_embed: torch.Tensor) -> torch.Tensor:
        h = self.transformer(response_embed)
        binary_logits = self.gate_proj(h)
        tau = 1.0 
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(binary_logits)))
        y = (binary_logits + gumbel_noise) / tau
        soft_gate = torch.sigmoid(y)
        hard_gate = (soft_gate > 0.5).float()
        binary_gate = hard_gate + (soft_gate - soft_gate.detach()) 
        binary_gate = binary_gate.squeeze(-1).squeeze(0)
        return binary_gate
    
class DOMA(ASRModel, ExportableEncDecModel, ASRModuleMixin, ASRBPEMixin):
    def __init__(self, cfg: DictConfig, trainer=None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
        self._setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = self.from_config_dict(self.cfg.preprocessor)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.encoder = AutoModel.from_pretrained(cfg.lm)
        self.ap = AdaptivePriorModule()
        self.decoder = self.from_config_dict(self.cfg.decoder)
        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None
        self.setup_optimization_flags()
        self.vocabulary = self.tokenizer.tokenizer.get_vocab()
        vocab_size = len(self.vocabulary)
        self.cfg.embedding["vocab_size"] = vocab_size
        self.embedding = self.from_config_dict(self.cfg.embedding)
        self.cfg.classifier["num_classes"] = vocab_size
        self.classifier = self.from_config_dict(self.cfg.classifier)
        self.loss = SmoothedNLLLoss(label_smoothing=self.cfg.loss.label_smoothing)
        self.sequence_generator = SequenceGenerator(
            cfg=self.cfg.sequence_generator,
            embedding=self.embedding,
            decoder=self.decoder,
            log_softmax=self.classifier,
            tokenizer=self.tokenizer,
        )
        decoding_cfg = self.cfg.get('decoding', None)
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg
        self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)
        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
            fold_consecutive=False,
        )

    def set_decoding_strategy(self, cfg: SequenceGeneratorConfig):
        cfg.max_sequence_length = self.sequence_generator.generator.max_seq_length
        self.sequence_generator = SequenceGenerator(cfg, self.embedding, self.decoder, self.classifier, self.tokenizer)

    @typecheck()
    def forward(
        self,
        oracle_asr=None,
        pred_asr=None,
    ):
        oracle_tokens = []
        N=5
        for oracle in oracle_asr:
            oracle_tokens.append(torch.tensor(LLADA_TOKENIZER(oracle)['input_ids']).to(LLADA_DEVICE))
        oracle_tokens = [x.squeeze(0) for x in oracle_tokens]
        oracle_tokens = pad_oracle_tokens(oracle_tokens, self.gen_length)

        loss = torch.tensor(0.0).to(next(self.encoder.parameters()).device)
        asr_inputs = pred_asr
        llada_asr_txt = []
        llada_asr_tokens = []
        for asr_input in asr_inputs:
            user_input=\
                f"""You are an expert assistant for automatic speech recognition systems. Your task is to check and improve ASR transcripts for intent recognition.
                    Input: an ASR transcript: "{asr_input}"
                    Prompt: Follow these rules to generate {N} ASR hypotheses:
                    1. Determine whether the transcript contains a plausible scenario, action, or entities rather than meaningless chit chat.
                    2. If the transcript conveys a plausible intent but has missing entities, fill in the missing entities based on semantic intent.
                    3. If the transcript does not express any request or intent, you may rewrite it reasonably.
                    4. Generate {N} hypotheses that consider all possible variations, including phonetically similar words.
                    5. If any word is unreasonable, replace it with a phonetically similar word rather than one similar in meaning.
                    6. Keep the sentence structure and word order intact as much as possible.
                    7. Use lowercase American English and ignore punctuation.
                    8. Output only the revised {N} hypotheses. Do not include explanations or commentary.
                    Output: the list of {N} revised ASR hypotheses
                    """
            m = [{"role": "user", "content": user_input}]
            user_input = LLADA_TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = LLADA_TOKENIZER(user_input)['input_ids']
            input_ids = torch.tensor(input_ids).to(LLADA_DEVICE).unsqueeze(0)
            prompt = input_ids
            out = generate(LLADA_MODEL, prompt, 
                        steps=32, gen_length=128, block_length=128, temperature=1.0, cfg_scale=0., remasking='low_confidence')
            out_nbest = LLADA_TOKENIZER.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            user_input =\
                f"""You are an expert assistant for automatic speech recognition systems. Your task is to detect and correct potential errors in the ASR transcript "{asr_input}".
                Input: the original ASR transcript: "{asr_input}"  
                Reference alternative transcriptions: {out_nbest}
                Prompt: Follow these rules to generate a corrected ASR transcript:
                1. Determine whether the transcript conveys a plausible scenario, action, or entities rather than meaningless chit chat.
                2. If the transcript conveys a plausible intent but has missing entities, fill in the missing entities based on semantic intent.
                3. If the transcript does not express any request or intent, you may rewrite it reasonably.
                4. If the transcript is already good, do not introduce unnecessary modifications.
                5. If any word is unreasonable, replace it with a phonetically similar word rather than one similar in meaning.
                6. Keep the sentence structure and word order intact as much as possible.
                7. Use lowercase American English and ignore punctuation.
                8. Output only the corrected ASR transcript. Do not include explanations or the n-best list.
                Output: the corrected ASR transcript
                """
            m = [{"role": "user", "content": user_input}]
            user_input = LLADA_TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = LLADA_TOKENIZER(user_input)['input_ids']
            input_ids = torch.tensor(input_ids).to(LLADA_DEVICE).unsqueeze(0)
            prompt = input_ids
            asr_ids = LLADA_TOKENIZER(asr_input)['input_ids']
            asr_ids = torch.tensor(asr_ids).to(LLADA_DEVICE).unsqueeze(0)
            asr = asr_ids
            out, loss_asr = generate_ap(LLADA_MODEL, LLADA_TOKENIZER, 
                        prompt, asr, None, self.ap,
                        steps=32, gen_length=64, block_length=64, 
                        temperature=1.0, cfg_scale=0., remasking='low_confidence',mode="train")
            loss += loss_asr
            out_txt = LLADA_TOKENIZER.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            out_txt = re.sub(r'[^\w\s]', '', out_txt).lower().strip()
            llada_asr_txt.append(out_txt)
            llada_asr_tokens.append(out)
        
        if len(llada_asr_tokens) == 0:
            llada_asr_tokens = torch.full((1, 1), 126081, dtype=torch.long, device=LLADA_DEVICE)
        else:
            llada_asr_tokens = [x.squeeze(0) for x in llada_asr_tokens]  
            llada_asr_tokens = pad_sequence(llada_asr_tokens, batch_first=True, padding_value=126081)
       
        return loss

    def training_step(self, batch, batch_nb):
        oracle_asr_texts = batch['oracle_asr'] 
        pred_asr_texts = batch['pred_asr']
        loss = self.forward(
            oracle_asr=oracle_asr_texts,
            pred_asr=pred_asr_texts,
        )
        tensorboard_logs = {
            'loss': loss.item(),
        }
        if len(self._optimizer.param_groups) == 1:
            tensorboard_logs['learning_rate'] = self._optimizer.param_groups[0]['lr']
        else:
            for i, group in enumerate(self._optimizer.param_groups):
                tensorboard_logs[f'learning_rate_g{i}'] = group['lr']
        return {'loss': loss, 'log': tensorboard_logs}

    def predict(
        self, asr_text=None
    ) -> List[str]:
        N=5
        llada_asr_txt = []
        asr_inputs = asr_text
        llada_asr_tokens = []
        for asr_input in asr_inputs:
            user_input=\
                f"""You are an expert in automatic speech recognition systems facing an ASR transcript for intent recognition.
                You should first determine whether the ASR contains a reasonable scenario/action/entities rather than meaningless chit chat.
                If the ASR conveys a plausible intent but has missing entities, you should use your imagination to complete it based on semantic intent.
                If the ASR does not appear to express any request, you may rewrite the sentence.
                Based on the original ASR input "{asr_input}", generate {N} different ASR hypotheses that consider all possible cases, including phonetically similar words.
                If there are unreasonable words, replace them with words that sound similar rather than similar in meaning.
                Important! The output must be in lowercase American English without punctuation. Return only the results. Do not include any explanations.
                """
            m = [{"role": "user", "content": user_input}]
            user_input = LLADA_TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = LLADA_TOKENIZER(user_input)['input_ids']
            input_ids = torch.tensor(input_ids).to(LLADA_DEVICE).unsqueeze(0)
            prompt = input_ids
            out = generate(LLADA_MODEL, prompt, 
                        steps=32, gen_length=128, block_length=128, temperature=1.0, cfg_scale=0., remasking='low_confidence')
            out_nbest = LLADA_TOKENIZER.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]

            user_input =\
                f"""You are an assistant for automatic speech recognition systems. 
                Your task is to detect and correct potential errors in the ASR transcript "{asr_text}" that lack a reasonable scenario/action/entities.
                Here are some reference alternative transcriptions: {out_nbest}.
                The transcriptions you generate must convey a reasonable intent rather than meaningless chit chat.
                If there are unreasonable words, replace them with words that sound similar rather than similar in meaning.
                If the ASR conveys a plausible intent but has missing entities, you should use your imagination to complete it based on semantic intent.
                If the ASR does not appear to express any request, you may rewrite the sentence.
                Ignore punctuation and use lowercase American English.
                Important! Do not provide any explanations or the n-best list. Return only the corrected asr transcription. 
                """
            m = [{"role": "user", "content": user_input}]
            user_input = LLADA_TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = LLADA_TOKENIZER(user_input)['input_ids']
            input_ids = torch.tensor(input_ids).to(LLADA_DEVICE).unsqueeze(0)
            prompt = input_ids
            asr_ids = LLADA_TOKENIZER(asr_input)['input_ids']
            asr_ids = torch.tensor(asr_ids).to(LLADA_DEVICE).unsqueeze(0)
            asr = asr_ids
            out, _ = generate_ap(LLADA_MODEL, LLADA_TOKENIZER, 
                        prompt, asr, None, self.ap,
                        steps=32, gen_length=64, block_length=64, temperature=1.0, cfg_scale=0., remasking='low_confidence')
            out_txt = LLADA_TOKENIZER.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            out_txt = re.sub(r'[^\w\s]', '', out_txt).lower().strip()
            llada_asr_txt.append(out_txt)
            llada_asr_tokens.append(out)
            
        tokens = self.bert_tokenizer(
            llada_asr_txt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        
        input_ids = tokens["input_ids"].to(next(self.encoder.parameters()).device)
        attention_mask = tokens["attention_mask"].to(next(self.encoder.parameters()).device)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded = outputs.last_hidden_state 
        encoded_mask = attention_mask
        pred_tokens = self.sequence_generator(encoded, encoded_mask)
        predictions = self.sequence_generator.decode_semantics_from_tokens(pred_tokens)

        return predictions

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        clean_signal, signal_len, semantics, semantics_len = batch['clean'][:4]
        oracle_asr_texts = batch['oracle_asr'] 
        pred_asr_texts = batch['pred_asr']
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, pred_len, predictions = self.forward(
                        oracle_asr=oracle_asr_texts,
                        pred_asr=pred_asr_texts,
                    )
        else:
            log_probs, pred_len, predictions = self.forward(
                        oracle_asr=oracle_asr_texts,
                        pred_asr=pred_asr_texts,
                    )

        eos_semantics = semantics[:, 1:]
        eos_semantics_len = semantics_len - 1  

        loss_value = self.loss(log_probs=log_probs, labels=eos_semantics, lengths=eos_semantics_len)

        self.wer.update(
            predictions=predictions,
            targets=eos_semantics,
            predictions_lengths=pred_len,
            targets_lengths=eos_semantics_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()

        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def test_dataloader(self):
        if self._test_dl is None:
            self._test_dl = []

        return self._test_dl

    def _setup_triplet_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_bpe_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False

        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = get_triplet_bpe_dataset(config=config, tokenizer=self.tokenizer)
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_bpe_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_bpe_dataset(
                config=config, tokenizer=self.tokenizer, augmentor=augmentor
            )
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_triplet_dataloader_from_config(config=train_data_config)

        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_triplet_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_triplet_dataloader_from_config(config=test_data_config)

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        if 'transcripts_filepath' in config:
            transcripts_filepath = config['transcripts_filepath']
        else:
            transcripts_filepath = os.path.join(config['temp_dir'], 'transcripts.json')

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'transcript_filepath': transcripts_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }
        
        temporary_datalayer = self._setup_triplet_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @torch.no_grad()
    def transcribe():
        pass   

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="slu_conformer_transformer_large_slurp",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:slu_conformer_transformer_large_slurp",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/slu_conformer_transformer_large_slurp/versions/1.13.0/files/slu_conformer_transformer_large_slurp.nemo",
        )
        results.append(model)

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer

