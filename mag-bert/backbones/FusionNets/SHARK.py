from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPooler
import torch.nn.functional as F
from torch import nn
import torch
from ..SubNets.AlignNets import AlignSubNet
from ..SubNets.FeatureNets import BERTEncoder, BertCrossEncoder

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

class SDIF(nn.Module):
    
    def __init__(self, args):

        super(SDIF, self).__init__()
        self.args = args
        # self.text_subnet = BERTEncoder.from_pretrained(args.text_backbone, cache_dir = args.cache_path)

        self.visual_size = args.video_feat_dim
        self.acoustic_size = args.audio_feat_dim
        self.text_size = args.text_feat_dim
        self.device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dst_feature_dims = args.dst_feature_dims

        self.layers_cross = args.n_levels_cross 
        self.layers_self = args.n_levels_self 

        self.dropout_rate = args.dropout_rate
        self.cross_dp_rate = args.cross_dp_rate
        self.cross_num_heads = args.cross_num_heads
        self.self_num_heads = args.self_num_heads

        # self.config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dst_feature_dims, nhead=self.self_num_heads)
        self.self_att = nn.TransformerEncoder(encoder_layer, num_layers=self.layers_self)

        self.video2text_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        self.audio2text_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        
        self.v2t_project = nn.Linear(self.visual_size, self.text_size)

        # self.mlp_project =  nn.Sequential(
        #         nn.Linear(self.dst_feature_dims, self.dst_feature_dims),
        #         nn.Dropout(args.dropout_rate),
        #         nn.GELU()
        #     )
        self.mlp_project =  nn.Sequential(
                nn.Linear(self.dst_feature_dims*3, self.dst_feature_dims),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

    def forward(self, text_feats, video_feats, audio_feats, text_mask):
        # first layer : T,V,A
        # bert_sent, bert_sent_mask, bert_sent_type = text_feats[:,0], text_feats[:,1], text_feats[:,2]
        bert_sent_mask = text_mask
        # text_outputs = self.text_subnet(text_feats)
        # text_seq, text_rep = text_outputs['last_hidden_state'], text_outputs['pooler_output']
        text_seq = text_feats  

        video_seq = self.v2t_project(video_feats)
        # video_seq = video_feats
        audio_seq = audio_feats

        video_mask = torch.sum(video_feats.ne(torch.zeros(video_feats[0].shape[-1]).to(self.device)).int(), dim=-1)/video_feats[0].shape[-1]
        video_mask_len = torch.sum(video_mask, dim=1, keepdim=True)  

        video_mask_len = torch.where(video_mask_len > 0.5, video_mask_len, torch.ones([1]).to(self.device))
        video_masked_output = torch.mul(video_mask.unsqueeze(2), video_seq)
        video_rep = torch.sum(video_masked_output, dim=1, keepdim=False) / video_mask_len
        
        audio_mask = torch.sum(audio_feats.ne(torch.zeros(audio_feats[0].shape[-1]).to(self.device)).int(), dim=-1)/audio_feats[0].shape[-1]
        audio_mask_len = torch.sum(audio_mask, dim=1, keepdim=True)  
        
        audio_masked_output = torch.mul(audio_mask.unsqueeze(2), audio_seq)
        audio_rep = torch.sum(audio_masked_output, dim=1, keepdim=False) / audio_mask_len
        
        # Second layer (V,A) --> T: V_T, A_T
        extended_video_mask = video_mask.unsqueeze(1).unsqueeze(2)
        extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype)
        extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        video2text_seq = self.video2text_cross(text_seq, video_seq, extended_video_mask)

        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0
        audio2text_seq = self.audio2text_cross(text_seq, audio_seq, extended_audio_mask)

        text_mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) 
        # print(bert_sent_mask.shape, bert_sent_mask.unsqueeze(2).shape, video2text_seq.shape)
        video2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), video2text_seq)
        video2text_rep = torch.sum(video2text_masked_output, dim=1, keepdim=False) / text_mask_len
        
        audio2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), audio2text_seq)
        audio2text_rep = torch.sum(audio2text_masked_output, dim=1, keepdim=False) / text_mask_len
        
        # Third layer: mlp->VAL
        # shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=1))
        shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=2))

        return shallow_seq

class Shark(nn.Module):
    def __init__(self, args):
        super(Shark, self).__init__()
        self.config = BertConfig.from_pretrained(args.text_backbone)
        self.text_encoder = TextEncoder(args)
        self.gate = GateModule(args)
        self.fushion = Fushion(args)

        # self.mag = MAG(self.config, args)
        self.sdif = SDIF(args)

        self.pooler = BertPooler(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)

        self.temp = args.temp

    def contrastive_loss(self, feats_1, feats_2):
        # print(feats_1.mean(dim=1).shape, feats_2.mean(dim=1).shape)
        feats_1 = feats_1.mean(dim=1)
        feats_2 = feats_2.mean(dim=1 )
        sim_matrix = torch.matmul(feats_1, feats_2.T)

        i_logsoftmax = nn.functional.log_softmax(sim_matrix / self.temp, dim=1)
        j_logsoftmax = nn.functional.log_softmax(sim_matrix.T / self.temp, dim=1)
        # print(i_logsoftmax.shape, i_logsoftmax.reshape([16,-1]).shape, i_logsoftmax.unsqueeze(1).shape)
        i_diag = torch.diag(i_logsoftmax)
        loss_i = i_diag.mean()

        j_diag = torch.diag(j_logsoftmax)
        loss_j = j_diag.mean()

        con_loss = - (loss_i + loss_j) / 2 #2024-07-18-11-04-57

        return con_loss
    
    def forward(self, text_feats, video_feats, audio_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats):
        text_mask = text_feats[:, 1]
        
        text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb = \
            self.text_encoder(text_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats)

        new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt = self.gate(text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb)

        text_feats = self.fushion(text_emb, new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt)
        # text_feats = self.fushion(text_emb, xReact_comet_emb, xWant_comet_emb) #remove sbert
        # text_feats = self.fushion(text_emb, xReact_sbert_emb, xWant_sbert_emb) #remove comet

        # output = torch.mean(self.mag(text_feats, video_feats, audio_feats), dim=1) #full model
        # output = self.pooler(text_feats) #remove audio&video
        # output = self.pooler(self.mag(text_feats, video_feats, audio_feats)) #full model
        output = self.pooler(self.sdif(text_feats, video_feats, audio_feats, text_mask))

        output = self.dropout(output)
        logits = self.classifier(output)
            
        # con_loss = self.contrastive_loss(new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt)
        return logits