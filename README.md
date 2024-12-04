# bangla_english_nmt_transformers

## 
```bash
python3 process_translation_datasets_and_push_dataset_to_hf.py \
    --input_file "/home/user/data.csv" \
    --output_file "/home/user/processed_data.csv" \
    --dataset_name "english_bangla_nmt_datasets" \
    --train_size 0.9 \
    --val_size 0.08 \
    --test_size 0.02 \
    --hf_token ""
```

