DATA_DIR="path_to_icsf_data"
CUDA_VISIBLE_DEVICES=0 python eval.py \
    dataset_manifest="${DATA_DIR}/test_icsf.json" \
    asr_transcripts_filepath="${DATA_DIR}/test_asr.json" \
    batch_size=16 \
    num_workers=8 \
    only_score_manifest=false \
    model_path="path_to_your_model" 

