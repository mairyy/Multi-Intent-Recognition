from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.SHARK import Shark
from .FusionNets.A3M import OurModel

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'shark': Shark,
    'a3m': OurModel,
}