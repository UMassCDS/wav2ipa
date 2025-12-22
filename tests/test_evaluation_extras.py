import pytest

import multipa.evaluation_extras


@pytest.mark.parametrize(
    "in_str,mapping,expected",
    [
        ("abc", {"ab": "x", "b": "y"}, "xc"),
        (
            "uʌɾ̃aɪheɪʔhiɹɪnmaɪsɛlfɑn",
            multipa.evaluation_extras.BUCKEYE_REDUCED_MAPPING,
            "uənaɪheɪʔhiɹɪnmaɪsɛlfɑn",
        ),
        (
            "jɪɾ̃oʊaɪaɪɾɪnhævɡʊɾɪkspiɹĩtsʌswɪθfɔstɹ̩hoʊmsɛn",
            multipa.evaluation_extras.BUCKEYE_REDUCED_MAPPING,
            "jɪnoʊaɪaɪɾɪnhævɡʊɾɪkspiɹitsəswɪθfɔstɹ̩hoʊmsɛn",
        ),
        (
            "aɪdseɪʌnãɪ̃ntĩsɛvɪniθɹi",
            multipa.evaluation_extras.BUCKEYE_REDUCED_MAPPING,
            "aɪdseɪənaɪntisɛvɪniθɹi",
        ),
        ("wɪθlɪmɪɾɪmʌɾ̃ĩ", multipa.evaluation_extras.BUCKEYE_REDUCED_MAPPING, "wɪθlɪmɪɾɪməni"),
    ],
)
def test_greedy_reduction_find_and_replace(in_str, mapping, expected):
    assert multipa.evaluation_extras.greedy_reduction_find_and_replace(in_str, mapping) == expected
