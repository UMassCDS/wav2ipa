# Evaluation

This folder contains evaluation notebook for evaluating multiple phonetic transcription models on the **TIMIT** dataset using **IPA (International Phonetic Alphabet)** transcriptions.

## Directory Structure

```text
Evaluation/
├── model_evaluation.ipynb                  # Main notebook to run and compare models on TIMIT
├── timit_to_ipa.py                         # Converts TIMIT .PHN files into IPA using phonecodes
├── Results/
│   ├── timit_dialect_model_comparison.csv              # Metrics per dialect group
│   ├── timit_model_evaluation_summary.csv              # Overall summary of model performance
│   └── timit_subset_with_actual_and_predictions.csv    # Subset predictions + ground truth
```

## Models Evaluated

- `ginic/data_seed_bs64_4_wav2vec2-large-xlsr-53-buckeye-ipa` (Ours - Multipa)
- `ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns` (Taguchi)
- `Allosaurus` (Model: `eng2102`)
- Whisper to Epitran pipelines
- `excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified` and `excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified` - output needs to be converted from the TIMIT symbol set to the IPA
- 