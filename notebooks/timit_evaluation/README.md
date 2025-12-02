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

## Data preparation
Obtain the [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1).
Preprocess it to get IPA transcriptions by running the `timit_to_ipa.py` script in this folder, passing the path to the COMPLETE folder in the TIMIT as an argument:
```
$ python timit_to_ipa.py your/path/to/timit/folder/COMPLETE
```

This will produce a file named complete_ipa.csv that with headers `audio_filename,ipa_transcription` that contains the relative path to the file in TIMIT and the IPA:
```
audio_filename,ipa_transcription
/COMPLETE/DR1/FAKS0/SA1.wav,ʃ i ɦ æ d j ɝ d ɑ ɹ k s u ɾ ɪ ŋ ɡ ɹ i s i w ɑ ʃ  w ɑ ɾ ɝ ʔ ɔ l j i ɚ
...
```

## Models Evaluated

- `ginic/data_seed_bs64_4_wav2vec2-large-xlsr-53-buckeye-ipa` (Ours - Multipa)
- `ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns` (Taguchi)
- `Allosaurus` (Model: `eng2102`)
- Whisper to Epitran pipelines
- `excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified` and `excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified` - output needs to be converted from the TIMIT symbol set to the IPA
-