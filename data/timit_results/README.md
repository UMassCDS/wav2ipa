# TIMIT Evaluation
This directory stores evaluation metrics and prediction details for models' performance on the *full TIMIT dataset*. The code for evaluation comes from `notebooks/timit_evaluation/model_evaluation.ipynb` and `notebooks/timit_evaluation/model_evaluation_espeak_only.ipynb`.

Note that although all prediction outputs were converted to IPA if needed (for excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified and excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k) and normalized to standard IPA unicode symbols using `ipatok.tokenise`, they were not post-processed to a reduced, shared symbol set.

The purposes of the subfolders are as follows:
- aggregate_metrics: "Leaderboard" CSVs that report aggregate (average for error rates, total sums for edit distances) results for each model. Although there are separate CSV files for different batches of experiments, all models are evaluated on the same full TIMIT dataset, so they can be combined into a single leaderboard. These files are produced by the `multipa-evaluate` script.
- edit_distances: Have detailed results on the most common phoneme substution, insertion and deletion errors that occur across test set samples for each model. There are 3 CSVs for each model, one for each kind of error. The naming template for files is `{model_name}_{error_type}.csv.`
- detailed_predictions: Contains one file per model with transcription predictions and metrics for every example in the test dataset, one example per line. Files are named like `{model_name}_detailed_predictions.csv`