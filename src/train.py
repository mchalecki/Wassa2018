import argparse
import logging
import time
from typing import Optional

from paths import DATA
from src.trainer import NetworkTrainer
from src.utils.types import path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train(train_path: path, test_path: path, model_name: Optional[str] = None) -> None:
    model = NetworkTrainer()
    log.info("Starting training")
    start = time.time()
    model.train(train_path, test_path, model_name)
    log.info(f"Training ended, took {start - time.time()}")
    model.export()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the wassa network')
    parser.add_argument('--train', '-tr', type=str, help='output checkpoint dir', metavar='TRAIN',
                        required=False, default=DATA.TRAIN.path)
    parser.add_argument('--test', '-te', type=str, help='test set path', metavar='TEST',
                        required=False, default=DATA.TEST.path)
    parser.add_argument('--name', '-n', type=str, help='name of directory where model will be stored', metavar='NAME',
                        required=False, default=None)
    return parser


def main() -> None:
    parser = create_parser()
    opts = parser.parse_args()
    train(opts.train, opts.test, opts.name)


if __name__ == '__main__':
    main()
