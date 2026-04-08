from __future__ import annotations

import pandas as pd

from seugearbox.features import build_feature_dataframe, rank_feature_importance


def test_feature_pipeline_produces_expected_columns() -> None:
    segmented = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 0],
            [2.0, 3.0, 4.0, 5.0, 1],
            [1.5, 2.5, 3.5, 4.5, 0],
            [2.5, 3.5, 4.5, 5.5, 1],
        ],
        columns=[0, 1, 2, 3, "labels"],
    )

    features = build_feature_dataframe(segmented, wavelet="db1", level=1)
    importance = rank_feature_importance(features)

    assert "故障类别" in features.columns
    assert len(features.columns) == 22
    assert list(importance.columns) == ["Feature", "Importance"]
