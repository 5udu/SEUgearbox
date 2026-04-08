from __future__ import annotations

EXPECTED_RAW_FILES = [
    "Health_20_0.csv",
    "Chipped_20_0.csv",
    "Miss_20_0.csv",
    "Root_20_0.csv",
    "Surface_20_0.csv",
    "ball_20_0.csv",
    "inner_20_0.csv",
    "outer_20_0.csv",
]

DEFAULT_FEATURE_COLUMNS = [
    "整流平均值",
    "均方根",
    "低频能量",
    "低频奇异值特征",
    "高频能量",
    "频带能量",
    "平均值",
    "均方频率",
    "重心频率",
]
DEFAULT_LABEL_COLUMN = "故障类别"

FALLBACK_FEATURE_COLUMNS = [
    "鏁存祦骞冲潎鍊?",
    "鍧囨柟鏍?",
    "浣庨鑳介噺",
    "浣庨濂囧紓鍊肩壒寰?",
    "楂橀鑳介噺",
    "棰戝甫鑳介噺",
    "骞冲潎鍊?",
    "鍧囨柟棰戠巼",
    "閲嶅績棰戠巼",
]
FALLBACK_LABEL_COLUMN = "鏁呴殰绫诲埆"

FEATURE_COLUMNS = [
    "最大值",
    "最小值",
    "平均值",
    "峰值",
    "峰峰值",
    "峭度",
    "偏度",
    "均方根",
    "整流平均值",
    "波形因子",
    "峰值因子",
    "裕度因子",
    "高频能量",
    "低频能量",
    "重心频率",
    "均方频率",
    "频率方差",
    "频带能量",
    "频率标准差",
    "高频奇异值特征",
    "低频奇异值特征",
]
