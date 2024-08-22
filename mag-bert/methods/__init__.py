from .MAG_BERT.manager import MAG_BERT
from .SHARK.manager import SHARK
from .TMT.manager import TMT

method_map = {
    'mag_bert': MAG_BERT,
    'shark': SHARK,
    'tmt': TMT
}
