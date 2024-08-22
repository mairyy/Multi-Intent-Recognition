from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.SHARK import Shark
from .FusionNets.TMT import TMT

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'shark': Shark,
    'tmt': TMT
}