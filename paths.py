from pathlib import Path

root = Path(__file__).resolve().parent


class DATA:
    path = root / 'data'

    class TRAIN:
        path = root / 'data' / 'train' / 'train-v3.csv'

    class TEST:
        path = root / 'data' / 'test' / 'test.csv'


class OUTPUTS:
    path = root / 'outputs'
