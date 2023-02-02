from __future__ import annotations

import os.path

DIRNAME = os.path.dirname(__file__)


def get_kytea_model_path() -> str:
    return DIRNAME + '/data/jp-0.4.7-5.mod'


def load_wagahaiwa_nekodearu() -> list[str]:
    with open(DIRNAME + '/data/wagahaiwa_nekodearu.txt') as fp:
        return list(fp)
