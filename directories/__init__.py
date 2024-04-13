import os

from pathlib import Path

root = Path(os.path.dirname(__file__)).parent

home = root.joinpath("re2g")

logging = root.joinpath("logging.yaml")

sqlite3 = root.joinpath("sqlite3.db")

fixtures = root.joinpath("fixtures")

korquad_v1_train = fixtures.joinpath("KorQuAD_v1.0_train.json")

korquad_v1_dev = fixtures.joinpath("KorQuAD_v1.0_dev.json")
