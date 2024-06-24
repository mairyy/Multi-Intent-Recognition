import torch
import logging
from torch import nn
from .__init__ import methods_map

__all__ = ['ModelManager']

class MIA(nn.Module):

    def __init__(self, args, tokenizer):

        super(MIA, self).__init__()

        self.args = args
        fusion_method = methods_map[args.method]
        self.model = fusion_method(args, tokenizer)

    def forward(self, text_feats, video_feats, audio_feats, video_ids=None, xReact_comet_feats=None, xWant_comet_feats=None, \
                xReact_sbert_feats=None, xWant_sbert_feats=None, training=True):

        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        
        if self.args.method == 'shark':
            mm_model = self.model(text_feats, video_feats, audio_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats)
        elif self.args.method == 'a3m':
            mm_model = self.model(text_feats, video_feats, audio_feats, video_ids, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats, training)
        else:
            mm_model = self.model(text_feats, video_feats, audio_feats)
        return mm_model
        
class ModelManager:

    def __init__(self, args, tokenizer):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.model = self._set_model(args, tokenizer)

    def _set_model(self, args, tokenizer):

        model = MIA(args, tokenizer) 
        model.to(self.device)
        return model