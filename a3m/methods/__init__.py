from .MAG_BERT.manager import MAG_BERT
from .SHARK.manager import SHARK
from .A3M.manager import A3M

method_map = {
    'mag_bert': MAG_BERT,
    'shark': SHARK,
    'a3m': A3M
}
