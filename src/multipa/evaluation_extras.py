"""Functions for evaluating non-HuggingFace models or models that need
additional post-processing steps to get desired standardized IPA outputs.

Note that you should have the 'dev' dependencies and may need additional external dependencies
such as https://github.com/festvox/flite and https://github.com/espeak-ng/espeak-ng to run this code.

Fair warning that this code is a hot mess and intended for evaluation and analysis purposes specific to this project.
Using it for more general purpose inference is not recommended.
"""

import tempfile

import datasets
from phonecodes import phonecodes
import soundfile as sf
import transformers
import tqdm


import multipa.data_utils
import multipa.evaluation


def allosaurus_predict(
    test_dataset, model="eng2102", phone_inventory="ipa", is_remove_spaces=True, is_normalize_ipa=False, num_proc=None
):
    import allosaurus.app
    import allosaurus.bin.download_model

    allosaurus_model_name = f"allosaurus_{model}_{phone_inventory}"
    allosaurus.bin.download_model.download_model(allosaurus_model_name)

    print("Evaluating allosaurus. Model:", model, "Phone inventory:", phone_inventory)
    model_predictions = []

    recog = allosaurus.app.read_recognizer(model)
    for audio in tqdm(test_dataset["audio"]):
        wav_path = audio["path"]
        data, sr = sf.read(wav_path)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, data, sr, format="WAV", subtype="PCM_16")
            prediction = recog.recognize(tmp.name, phone_inventory)
            # prediction = model.recognize(audio["path"], phone_inventory)
            model_predictions.append({multipa.evaluation.PREDICTION_KEY: prediction})
    predictions_dataset = datasets.Dataset.from_list(model_predictions)
    predictions_dataset = predictions_dataset.map(
        lambda x: multipa.data_utils.clean_text(
            x, text_key=multipa.evaluation.PREDICTION_KEY, is_remove_space=is_remove_spaces, is_normalize_ipa=is_normalize_ipa
        ),
        num_proc=num_proc,
    )
    return predictions_dataset


def hf_model_to_epitran_predict(
    model_name, test_dataset, device, num_proc=None, is_remove_spaces=True, is_normalize_ipa=False
):
    import epitran

    print("Building pipeline and downloading model")
    if model_name.endswith(".en"):
        pipe = transformers.pipeline("automatic-speech-recognition", model=model_name, device=device)
    else:
        pipe = transformers.pipeline(
            "automatic-speech-recognition", model=model_name, generate_kwargs={"language": "english"}, device=device
        )
    print("Predicting with", model_name)
    orthography_predictions = [d["text"] for d in pipe(test_dataset["audio"])]
    epi = epitran.Epitran("eng-Latn")
    print("Transliterating with Epitran")
    ipa_predictions = []
    for pred in tqdm(orthography_predictions):
        result = epi.transliterate(pred)
        ipa_predictions.append({multipa.evaluation.PREDICTION_KEY: result})
    predictions_dataset = datasets.Dataset.from_list(ipa_predictions)
    predictions_dataset = predictions_dataset.map(
        lambda x: multipa.data_utils.clean_text(
            x, text_key=multipa.evaluation.PREDICTION_KEY, is_remove_space=is_remove_spaces, is_normalize_ipa=is_normalize_ipa
        ),
        num_proc=num_proc,
    )
    return predictions_dataset


def phonecodes_convert_batch(batch: dict, in_code="timit", out_code="ipa", post_conversion_mapping=None):
    """
    Phonecodes conversion that operates on Datasets
    """
    in_str = batch[in_code]
    conversion = phonecodes.convert(in_str, in_code, out_code, post_conversion_mapping=post_conversion_mapping)
    batch[out_code] = conversion
    return batch


def hf_to_phonecodes(
    test_dataset,
    model_name="excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k",
    in_code="timit",
    out_code="ipa",
    ipa_post_processing_map=None,
    device=None,
    num_proc=None,
    is_remove_spaces=True,
    is_normalize_ipa=False,
):
    pipe = transformers.pipeline("automatic-speech-recognition", model=model_name, device=device)
    predictions_dataset = datasets.Dataset.from_list([{in_code: d["text"]} for d in pipe(test_dataset["audio"])])
    # convert to ipa
    predictions_dataset = predictions_dataset.map(
        lambda x: phonecodes_convert_batch(x, in_code, out_code, post_conversion_mapping=ipa_post_processing_map),
        num_proc=device,
    )
    # clean prediction output
    predictions_dataset = predictions_dataset.map(
        lambda x: multipa.data_utils.clean_text(
            x, text_key=out_code, is_remove_space=is_remove_spaces, is_normalize_ipa=is_normalize_ipa
        ),
        num_proc=num_proc,
    )
    predictions_dataset = predictions_dataset.rename_column(out_code, multipa.evaluation.PREDICTION_KEY)
    predictions_dataset = predictions_dataset.rename_column(in_code, f"{in_code}_{multipa.evaluation.PREDICTION_KEY}")
    return predictions_dataset
