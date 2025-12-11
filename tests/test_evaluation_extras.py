import pytest

from phonecodes import phonecodes

import multipa.evaluation_extras


@pytest.mark.parametrize(
    "in_str,mapping,expected",
    [
        ("abc", {"ab": "x", "b": "y"}, "xc"),
        (
            "uʌɾ̃aɪheɪʔhiɹɪnmaɪsɛlfɑn",
            phonecodes.phonecode_tables.BUCKEYE_IPA_TO_TIMIT_BUCKEYE_SHARED,
            "uənaɪheɪʔhiɹɪnmaɪsɛlfɑn",
        ),
    ],
)
def test_greedy_reduction_find_and_replace(in_str, mapping, expected):
    assert multipa.evaluation_extras.greedy_reduction_find_and_replace(in_str, mapping) == expected
