import os.path

DIRNAME = os.path.dirname(__file__)


def get_kytea_model_path():
    return DIRNAME + '/data/jp-0.4.7-6.mod'


def load_wagahaiwa_nekodearu():
    with open(DIRNAME + '/data/wagahaiwa_nekodearu.txt') as fp:
        return fp.read()
