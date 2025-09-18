import json
from pathlib import Path
from torch.utils.data import Dataset
from omegaconf import ListConfig
from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.parts.preprocessing.features import AudioAugmentor

def load_asr_transcript(filepaths):
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    elif isinstance(filepaths, ListConfig):
        filepaths = list(filepaths)

    transcript_dict = {}
    for filepath in filepaths:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Transcript file not found: {filepath}")
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)
                utt_id = line_data["id"]
                predicted_text = line_data.get("predicted_text_cleaned", "")
                ground_truth_text = line_data.get("ground_truth_cleaned", "")
                transcript_dict[utt_id] = (predicted_text, ground_truth_text)
    return transcript_dict


def load_slu_labels(manifest_file):
    if isinstance(manifest_file, ListConfig):
        manifest_file = list(manifest_file)
    if isinstance(manifest_file, list):
        files = manifest_file
    else:
        files = [manifest_file]

    slu_dict = {}
    for fpath in files:
        fpath = Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"Manifest file not found: {fpath}")
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)
                utt_id = line_data["id"]
                semantics = line_data.get("semantics", {})
                slu_dict[utt_id] = semantics
    return slu_dict


class PairedBPEAudioDataset(Dataset):
    def __init__(
        self,
        clean_manifest: str,
        tokenizer,
        sample_rate: int,
        clean_asr_path: str,
        int_values: bool = False,
        augmentor: AudioAugmentor = None,
        max_duration=None,
        min_duration=None,
        max_utts=0,
        trim=False,
        use_start_end_token=True,
        return_sample_id=False,
        channel_selector=None,
    ):
        super().__init__()

        self.clean_dataset = audio_to_text.AudioToBPEDataset(
            manifest_filepath=clean_manifest,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=None,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            use_start_end_token=use_start_end_token,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
        )
        self.clean_asr = load_asr_transcript(clean_asr_path)
        if return_sample_id:
            assert len(self.clean_dataset) == len(self.clean_asr), "Mismatch between clean manifest and clean ASR"
            
        self.return_sample_id = return_sample_id

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        clean_sample = self.clean_dataset[idx]
        sample_id = str(clean_sample[4])
        clean_pred_text, clean_gt_text = self.clean_asr[str(sample_id)]
        clean_label = self.clean_slu[str(sample_id)]
        data = {
            'clean': clean_sample,
            'clean_asr_pred': clean_pred_text,
            'clean_asr_gt': clean_gt_text,
            'clean_slu_label': clean_label,
        }
        return data

    @property
    def collate_fn(self):
        return paired_collate_fn


def paired_collate_fn(batch):
    clean_batch = [item['clean'] for item in batch]
    clean_pred_texts = [item['clean_asr_pred'] for item in batch]
    clean_gt_texts = [item['clean_asr_gt'] for item in batch]
    clean_collated = audio_to_text._speech_collate_fn(clean_batch, pad_id=0)
    return {
        'clean': clean_collated,
        'pred_asr': clean_pred_texts,
        'oracle_asr': clean_gt_texts,
    }


def get_triplet_bpe_dataset(
    config: dict,
    tokenizer,
    augmentor: AudioAugmentor = None,
) -> PairedBPEAudioDataset:
    return PairedBPEAudioDataset(
        clean_manifest=config['manifest_filepath'],
        clean_asr_path=config['transcript_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
