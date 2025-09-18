DATA_DIR="path_to_icsf_data"
EXP_NAME="your_exp_name"  
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-path="./configs" --config-name=config \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_icsf.json]" \
    model.train_ds.transcript_filepath="[${DATA_DIR}/train_asr.json]" \
    model.validation_ds.manifest_filepath="[${DATA_DIR}/devel_icsf.json]" \
    model.validation_ds.transcript_filepath="[${DATA_DIR}/devel_asr.json]" \
    model.test_ds.manifest_filepath="[${DATA_DIR}/test_icsf.json]" \
    model.test_ds.transcript_filepath="[${DATA_DIR}/test_icsf.json]" \
    model.tokenizer.dir="${DATA_DIR}/your_path_to_icsf_tokenizer" \
    model.train_ds.batch_size=16 \
    model.validation_ds.batch_size=16 \
    model.test_ds.batch_size=16 \
    model.ckpt_dir="./experiments/${EXP_NAME}/ckpt/" \
    trainer.devices=1 \
    trainer.max_epochs=10 \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.name="${EXP_NAME}" \

    

