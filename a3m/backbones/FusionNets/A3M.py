from transformers import BertModel, BertConfig, BartModel
from transformers.models.bert.modeling_bert import BertPooler
import torch.nn.functional as F
from torch import nn
import torch
from typing import Optional
import numpy as np

from ..SubNets.AlignNets import AlignSubNet
from ..SubNets.resnet.resnet_utils import myResnet
from ..SubNets.resnet import resnet

class TextEncoder(nn.Module):
    def __init__(self, args, tokenizer):
        super(TextEncoder, self).__init__()
        self.method = args.method
        self.encoder = BertModel.from_pretrained(args.text_backbone)
        self.encoder.resize_token_embeddings(len(tokenizer))

    def get_token_emb(self, text, video):
        input_ids = torch.cat([video[:,0], text[:,0]], dim=1)
        attention_mask = torch.cat([video[:,1], text[:,1]], dim=1)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        padding = torch.zeros([len(input_ids), len(input_ids[0]) - len(text[:,3][0])])
        noun_masks = torch.cat([padding, text[:,3]], dim=1)
        token_embeddings = outputs.last_hidden_state

        return token_embeddings, noun_masks, token_embeddings[:,video[:,1].shape[1]:]
    
    def get_embedding(self, text):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state  #torch.size([batch_size, n_Tokens, text_feat_dim])
        sequence_embedding = torch.mean(word_embeddings, dim=1) #torch.size([batch_size, text_feat_dim])    

        return word_embeddings
    
    def forward(self, text_feats, xReact_comet, xWant_comet, xReact_sbert, xWant_sbert, video_ids):
        xReact_comet_emb = self.get_embedding(xReact_comet)
        xWant_comet_emb = self.get_embedding(xWant_comet)
        xReact_sbert_emb = self.get_embedding(xReact_sbert)
        xWant_sbert_emb = self.get_embedding(xWant_sbert)
        token_emb, noun_masks, text_emb = self.get_token_emb(text_feats, video_ids)

        return xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb,  token_emb, noun_masks, text_emb
    
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

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.text_feat_dim
        
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
        # print(visual.shape, acoustic.shape)
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

        return embedding_output
    
class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim):
        super(ImageEmbedding, self).__init__()
        self.linear = nn.Linear(image_dim, final_dim)

    def forward(self, image_features):
        img_len = list(map(len, image_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, image_features))

        embedded = None
        if len(non_empty_features) > 0:
            img_tensor = torch.cat(non_empty_features, dim=0)
            embedded = self.linear(img_tensor)

        output = []
        index = 0
        for l in img_len:
            if l > 0:
                output.append(embedded[index:index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output

class A3M(nn.Module):
    def __init__(self, args, tokenizer):
        super(A3M, self).__init__()

        self.noun_linear = nn.Linear(768,768)
        self.multi_linear = nn.Linear(768,768)
        self.att_linear = nn.Linear(768*2,1)
        self.linear=nn.Linear(768*2,1)
        self.alpha_linear1=nn.Linear(768,768)
        self.alpha_linear2=nn.Linear(768,768)

    def _embed_multi_modal(self, token_embed, image_embed):
        """
        :param token_embed: Tensor, [batch_size, 81, 768]
        :param image_embed: List of Tensor, [49, 768]
        :return:
            token_embed: concate text and visual into one embedding
        """
        if not image_embed[0].dtype == torch.float32:
            token_embed = token_embed.half()

        for index, value in enumerate(image_embed):
            for i in range(len(value)):
                if i != 0:
                    token_embed[index, i] = value[i]
          
        return token_embed
    
    def get_noun_embed(self, feature, noun_masks):
        noun_masks = noun_masks.cpu()
        noun_num = [x.numpy().tolist().count(1) for x in noun_masks]
        noun_position=[np.where(np.array(x)==1)[0].tolist() for x in noun_masks]
        for i,x in enumerate(noun_position):
            assert len(x)==noun_num[i]
        max_noun_num = max(noun_num)

        # pad
        for i,x in enumerate(noun_position):
            if len(x)<max_noun_num:
                noun_position[i]+=[0]*(max_noun_num-len(x))
        noun_position=torch.tensor(noun_position)
        noun_embed=torch.zeros(feature.shape[0],max_noun_num,feature.shape[-1])
        for i in range(len(feature)):
            if noun_position[i].dtype != torch.int32 and noun_position[i].dtype != torch.int64:
                continue
            else:
                noun_embed[i]=torch.index_select(feature[i],dim=0,index=noun_position[i])
        return noun_embed
    
    def noun_attention(self, encoder_outputs, noun_embed):
        multi_features_rep = encoder_outputs.unsqueeze(2).repeat(1, 1, noun_embed.shape[1], 1)
        noun_features_rep = noun_embed.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1, 1)
        noun_features_rep = self.noun_linear(noun_features_rep)
        multi_features_rep = self.multi_linear(multi_features_rep)
        concat_features = torch.tanh(torch.cat([noun_features_rep, multi_features_rep], dim=-1))
        att = torch.softmax(self.att_linear(concat_features).squeeze(-1), dim=-1)
        att_features = torch.matmul(att, noun_embed)

        alpha = torch.sigmoid(self.linear(torch.cat([self.alpha_linear1(encoder_outputs), self.alpha_linear2(att_features)], dim=-1)))
        alpha = alpha.repeat(1, 1, 768)

        encoder_outputs = torch.mul(1-alpha, encoder_outputs) + torch.mul(alpha, att_features)

        return encoder_outputs
    
    def forward(self, token_emb, video_emb, noun_masks):
        token_feats = self._embed_multi_modal(token_emb, video_emb)
        noun_embed = self.get_noun_embed(token_feats, noun_masks)
        visual_feats = self.noun_attention(token_feats, noun_embed)

        return visual_feats
    
class OurModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(OurModel, self).__init__()
        self.config = BertConfig.from_pretrained(args.text_backbone)
        self.text_encoder = TextEncoder(args, tokenizer)
        self.gate = GateModule(args)
        self.fushion = Fushion(args)

        net = getattr(resnet, 'resnet152')()
        net.load_state_dict(torch.load('/Users/admin/Documents/Projects/Multi-Intent-Recognition/a3m/backbones/FusionNets/resnet152-b121ed2d.pth'))
        self.img_encoder = myResnet(net, True, args.device)
        self.img_encoder.to(args.device)
        self.image_emb = ImageEmbedding(2048, args.text_feat_dim)

        self.a3m = A3M(args, tokenizer)

        self.mag = MAG(self.config, args)

        self.pooler = BertPooler(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)

    def forward(self, text_feats, video_feats, audio_feats, video_ids, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats):
        xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb, token_emb, noun_masks, text_emb = \
            self.text_encoder(text_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats, video_ids)

        new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt = self.gate(text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb)

        text_feats = self.fushion(text_emb, new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt)
        
        imgs_f, img_mean, img_att = self.img_encoder(video_feats)
        video_feats = img_att.view(-1, 2048, 49).permute(0, 2, 1)
        video_feats = self.image_emb(video_feats)

        video_feats = self.a3m(token_emb, video_feats, noun_masks)
        
        # output = self.pooler(torch.cat([text_feats, video_feats, audio_feats], dim=1))
        output = self.pooler(self.mag(text_feats, video_feats, audio_feats))

        output = self.dropout(output)
        logits = self.classifier(output)
            
        return logits