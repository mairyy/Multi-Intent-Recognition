from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.SDIF import SDIF

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'sdif': SDIF,
}