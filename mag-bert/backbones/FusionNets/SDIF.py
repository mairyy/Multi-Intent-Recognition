import torch
from ..SubNets.FeatureNets import BERTEncoder, BertCrossEncoder
from torch import nn
from transformers import BertConfig
from ..SubNets.AlignNets import AlignSubNet
import torch.nn.functional as F

__all__ = ['SDIF']

class MAG(nn.Module):
    def __init__(self,  config, args):
        super(MAG, self).__init__()
        self.args = args

        if self.args.need_aligned:
            self.alignNet = AlignSubNet(args, args.mag_aligned_method)

        # text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        feat_dim = args.text_feat_dim
        
        self.W_hv = nn.Linear(feat_dim*2, feat_dim)
        self.W_ha = nn.Linear(feat_dim*2, feat_dim)

        self.W_v = nn.Linear(feat_dim, feat_dim)
        self.W_a = nn.Linear(feat_dim, feat_dim)

        self.beta_shift = args.beta_shift

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        if self.args.need_aligned:
            text_embedding, visual, acoustic  = self.alignNet(text_embedding, visual, acoustic)
    
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embedding.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embedding.device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output

class SDIF(nn.Module):
    
    def __init__(self, args):

        super(SDIF, self).__init__()
        self.args = args
        self.text_subnet = BERTEncoder.from_pretrained(args.text_backbone, cache_dir = args.cache_path)

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

        self.config = BertConfig.from_pretrained("bert-base-uncased")
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
        
        self.MAG = MAG(self.config, args)

        self.linear =  nn.Sequential(
                nn.Linear(self.dst_feature_dims*2, self.dst_feature_dims),
                nn.Dropout(args.dropout_rate)
            )
        
        self.shallow_att_project = nn.Linear(self.dst_feature_dims, 1, bias=False)
        self.deep_att_project = nn.Linear(self.dst_feature_dims, 1, bias=False)

        self.activation = nn.ReLU()
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.dst_feature_dims, out_features=self.dst_feature_dims))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.dst_feature_dims, out_features= args.num_labels))

        if args.aug:
            self.out_layer = nn.Linear(self.dst_feature_dims, args.num_labels)
            self.aug_dp = nn.Dropout(args.aug_dp)


    def forward(self, text_feats, video_feats, audio_feats):
        # first layer : T,V,A
        bert_sent, bert_sent_mask, bert_sent_type = text_feats[:,0], text_feats[:,1], text_feats[:,2]
        text_outputs = self.text_subnet(text_feats)
        text_seq, text_rep = text_outputs['last_hidden_state'], text_outputs['pooler_output']
        

        video_seq = self.v2t_project(video_feats)
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

        video2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), video2text_seq)
        video2text_rep = torch.sum(video2text_masked_output, dim=1, keepdim=False) / text_mask_len
        

        audio2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), audio2text_seq)
        audio2text_rep = torch.sum(audio2text_masked_output, dim=1, keepdim=False) / text_mask_len
        

        # Third layer: mlp->VAL
        # shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=1))
        shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=2))
        
        # Deep Interaction
        # tri_cat_mask = torch.cat([bert_sent_mask, bert_sent_mask, bert_sent_mask], dim=-1)

        # tri_mask_len = torch.sum(tri_cat_mask, dim=1, keepdim=True) 
        # shallow_masked_output = torch.mul(tri_cat_mask.unsqueeze(2), shallow_seq)
        # shallow_rep = torch.sum(shallow_masked_output, dim=1, keepdim=False) / tri_mask_len

        # all_reps = torch.stack((text_rep, video_rep, audio_rep, video2text_rep, audio2text_rep, shallow_rep), dim=0)
        # all_hiddens = self.self_att(all_reps)
        # deep_rep = torch.cat((all_hiddens[0], all_hiddens[1], all_hiddens[2], all_hiddens[3], all_hiddens[4], all_hiddens[5]), dim=1)

        # self.text_rep = text_rep
        # self.video_rep = video_rep
        # self.audio_rep = audio_rep
        # self.video2text_rep = video2text_rep
        # self.audio2text_rep = audio2text_rep
        # self.shallow_rep = shallow_rep

        # logits = deep_rep #utput shape:  torch.Size([8, 4608]) 
        # logits = self.fusion(deep_rep) #utput shape:  torch.Size([8, 20])
        # logits = shallow_seq #utput shape:  torch.Size([8, 90, 768]) => torch.Size([8, 30, 768])

        #fushion = MAG   
        z_seq = self.linear(torch.cat([text_seq, shallow_seq], dim=2)) #torch.Size([8, 30, 1536]) => torch.Size([8, 30, 768])
        z_rep = torch.sum(z_seq, dim=1) / z_seq.shape[1] #torch.Size([8, 768]) 
        outputs = self.MAG(z_rep, video_rep, audio_rep)
        logits = self.fusion(outputs)

        return logits
    
    def get_rep(self, text_feats, video_feats, audio_feats):
        logits = self.forward(text_feats, video_feats.float(), audio_feats.float())
        text_rep, video_rep, audio_rep, video2text_rep, audio2text_rep, shallow_rep = self.text_rep, self.video_rep, self.audio_rep, self.video2text_rep, self.audio2text_rep, self.shallow_rep
        return text_rep, video_rep, audio_rep, video2text_rep, audio2text_rep, shallow_rep
    
    def pre_train(self, text_feats):
        text_outputs = self.text_subnet(text_feats)
        text_seq, text_rep = text_outputs['last_hidden_state'], text_outputs['pooler_output']
        text_rep = self.aug_dp(text_rep)
        logits = self.out_layer(text_rep)
        return logits


    

