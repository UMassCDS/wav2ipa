"""Functions for evaluating non-HuggingFace models or models that need
additional post-processing steps to get desired standardized IPA outputs.

Note that you should have the 'dev' dependencies and may need additional external dependencies
such as https://github.com/festvox/flite and https://github.com/espeak-ng/espeak-ng to run this code.

Fair warning that this code is a hot mess and intended for evaluation and analysis purposes specific to this project.
Using it for more general purpose inference is not recommended.
"""

import re
import tempfile
from typing import Literal

import datasets
from phonecodes import phonecodes
import soundfile as sf
import transformers
from tqdm import tqdm


import multipa.data_utils
import multipa.evaluation

# Post processing map for TIMIT dataset and models not trained on Buckeye
TIMIT_AND_OTHER_REDUCED_MAPPING = phonecodes.phonecode_tables.TIMIT_IPA_TO_TIMIT_BUCKEYE_SHARED
TIMIT_AND_OTHER_REDUCED_MAPPING["ː"] = ""
TIMIT_AND_OTHER_REDUCED_MAPPING["ũ"] = "u"  # Emitted by facebook model literally once

# There are extra versions of nasalized characters that are single symbols instead of the +diacritic versions that appear often
BUCKEYE_REDUCED_MAPPING = phonecodes.phonecode_tables.BUCKEYE_IPA_TO_TIMIT_BUCKEYE_SHARED
BUCKEYE_REDUCED_MAPPING["õ"] = "o"
BUCKEYE_REDUCED_MAPPING["ĩ"] = "i"
BUCKEYE_REDUCED_MAPPING["ã"] = "a"


def allosaurus_predict(
    test_dataset, model="eng2102", phone_inventory="ipa", is_remove_spaces=True, is_normalize_ipa=False, num_proc=None
):
    import allosaurus.app
    import allosaurus.bin.download_model

    allosaurus.bin.download_model.download_model(model)

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
        num_proc=num_proc,
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


def get_reduction_pattern(reduction_mapping: dict[str, str]):
    return "|".join(re.escape(k) for k in reduction_mapping.keys())


def dataset_reduction_greedy_find_and_replace(
    dataset: datasets.Dataset,
    column: str,
    ipa_source: Literal["buckeye", "timit"],
    num_proc: int,
    custom_reduction_mapping: dict[str, str] | None = None,
):
    """Convert symbols in entries from dataset column according to the specified reduction mapping using a
    greedy find and replace.

    Args:
        dataset: Input dataset
        column: Column in which to perform symbol replacements
        ipa_source: Choose from a predefined "buckeye" or "timit" reduction mapping.
        num_proc: How many processes to use when running mapping
        custom_reduction_mapping: Set to override reduction mapping looked up using the ipa_source param. Defaults to None

    Returns:
        The dataset with symbols in the specified column replaced according to the reduction mapping.
    """

    if ipa_source.lower() == "buckeye":
        reduction_mapping = BUCKEYE_REDUCED_MAPPING
    elif ipa_source.lower() == "timit":
        reduction_mapping = TIMIT_AND_OTHER_REDUCED_MAPPING
    elif custom_reduction_mapping is not None:
        reduction_mapping = custom_reduction_mapping
    else:
        raise ValueError("You must specify a non-empty reduction mapping")

    pattern = get_reduction_pattern(reduction_mapping)

    # Nested function to find and replace strings in each row of the dataset
    def batch_find_and_replace(batch):
        input_string = batch[column]
        replacement_str = re.sub(pattern, lambda match: reduction_mapping[match.group()], input_string)
        batch[column] = replacement_str
        return batch

    return dataset.map(
        lambda x: batch_find_and_replace(x),
        num_proc=num_proc,
    )


def greedy_reduction_find_and_replace(input_string: str, reduction_mapping: dict[str, str]):
    """Greedy find and replace an ordered symbol mapping (reduction).
    This is just another version of the post_process_reduction function that works directly on strings and
    can be used easily in numpy or pandas data structures.

    Args:
        input_string: String in which to replace symbols
        reduction_mapping: Custom ordered mapping for greedy replacement search

    Returns:
        A copy of the original string with any symbols replaced according to the mapping
    """
    try:
        replacement_str = re.sub(
            get_reduction_pattern(reduction_mapping), lambda match: reduction_mapping[match.group()], input_string
        )
        return replacement_str
    except TypeError:
        raise TypeError(f"Expected string or bytes-like object, got '{input_string}' with type '{type(input_string)}'")
