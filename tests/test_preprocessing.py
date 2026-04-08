from __future__ import annotations

import pandas as pd

from seugearbox.preprocessing import segment_signal_frame


def test_segment_signal_frame_builds_expected_windows() -> None:
    raw = pd.DataFrame(
        {
            "class_a": [1, 2, 3, 4, 5, 6],
            "class_b": [10, 11, 12, 13, 14, 15],
        }
    )

    segmented = segment_signal_frame(raw, window_size=3, step_size=2)

    assert segmented.shape == (4, 4)
    assert segmented["labels"].tolist() == [0, 0, 1, 1]
