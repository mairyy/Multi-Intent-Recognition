from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPooler
import torch.nn.functional as F
from torch import nn
import torch
from ..SubNets.AlignNets import AlignSubNet

class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(args.text_backbone)

    def get_embedding(self, text):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state  #torch.size([batch_size, n_Tokens, text_feat_dim])
        sequence_embedding = torch.mean(word_embeddings, dim=1) #torch.size([batch_size, text_feat_dim])

        return word_embeddings
    
    def forward(self, text_feats, xReact_comet, xWant_comet, xReact_sbert, xWant_sbert):
        text_emb = self.get_embedding(text_feats)
        #remove comet
        # xReact_comet_emb = xReact_comet
        # xWant_comet_emb = xWant_comet
        xReact_comet_emb = self.get_embedding(xReact_comet)
        xWant_comet_emb = self.get_embedding(xWant_comet)
        xReact_sbert_emb = self.get_embedding(xReact_sbert)
        xWant_sbert_emb = self.get_embedding(xWant_sbert)
        #remove sbert
        # xReact_sbert_emb = xReact_sbert
        # xWant_sbert_emb = xWant_sbert
        return text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb
    
class GateModule(nn.Module):
    def __init__(self, args):
        super(GateModule, self).__init__()
        self.linear_layer = nn.Linear(args.text_feat_dim + args.relation_feat_dim * 2, args.text_feat_dim)
    
    def forward(self, text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb):
        xReact_encoder_outputs_utt = torch.cat((text_emb, xReact_comet_emb, xReact_sbert_emb), -1)
        xWant_encoder_outputs_utt = torch.cat((text_emb, xWant_comet_emb, xWant_sbert_emb), -1)
        indicator_r = self.linear_layer(xReact_encoder_outputs_utt)
        indicator_w = self.linear_layer(xWant_encoder_outputs_utt)
        
        indicator_r_ = F.softmax(indicator_r, dim=-1)
        indicator_w_ = F.softmax(indicator_w, dim=-1)

        indicator_r_ = indicator_r_[:, :, 0].unsqueeze(2).repeat(1, 1, text_emb.size(-1))
        indicator_w_ = indicator_w_[:, :, 0].unsqueeze(2).repeat(1, 1, text_emb.size(-1))

        new_xReact_encoder_outputs_utt = indicator_r_ * xReact_comet_emb + (1 - indicator_r_) * xReact_sbert_emb
        new_xWant_encoder_outputs_utt = indicator_w_ * xWant_comet_emb + (1 - indicator_w_) * xWant_sbert_emb

        return new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt
    
class Fushion(nn.Module):
    def __init__(self, args):
        super(Fushion, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(args.text_feat_dim))
        self.W.requires_grad = True
        self.alpha = args.weight_fuse_relation

    def forward(self, text_emb, xReact_emb, xWant_emb):
        z1 = text_emb + self.W * xReact_emb
        z2 = text_emb + self.W * xWant_emb

        z = self.alpha * z1 + (1 - self.alpha) * z2

        return z
    
class MAG(nn.Module):
    def __init__(self,  config, args):
        super(MAG, self).__init__()
        self.args = args

        if self.args.need_aligned:
            self.alignNet = AlignSubNet(args, args.aligned_method)

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        
        self.W_hv = nn.Linear(video_feat_dim + text_feat_dim, text_feat_dim)
        self.W_ha = nn.Linear(audio_feat_dim + text_feat_dim, text_feat_dim)

        self.W_v = nn.Linear(video_feat_dim, text_feat_dim)
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim)

        self.beta_shift = args.beta_shift

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        if self.args.need_aligned:
            text_embedding, acoustic, visual  = self.alignNet(text_embedding, acoustic, visual)
        
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(self.args.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(self.args.device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        # return torch.mean(embedding_output, dim=1)
        return embedding_output
    
class Shark(nn.Module):
    def __init__(self, args):
        super(Shark, self).__init__()
        self.config = BertConfig.from_pretrained(args.text_backbone)
        self.text_encoder = TextEncoder(args)
        self.gate = GateModule(args)
        self.fushion = Fushion(args)
        self.mag = MAG(self.config, args)
        self.pooler = BertPooler(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)

    def forward(self, text_feats, video_feats, audio_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats):
        text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb = \
            self.text_encoder(text_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats)

        new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt = self.gate(text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb)

        text_feats = self.fushion(text_emb, new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt)
        # text_feats = self.fushion(text_emb, xReact_comet_emb, xWant_comet_emb) #remove sbert
        # text_feats = self.fushion(text_emb, xReact_sbert_emb, xWant_sbert_emb) #remove comet

        # output = torch.mean(self.mag(text_feats, video_feats, audio_feats), dim=1) #full model
        # output = self.pooler(text_feats) #remove audio&video
        output = self.pooler(self.mag(text_feats, video_feats, audio_feats)) #full model

        output = self.dropout(output)
        logits = self.classifier(output)
            
        return logits