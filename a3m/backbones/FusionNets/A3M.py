import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartPreTrainedModel, BartConfig, BartTokenizer, BertModel, AutoTokenizer
import json
import random
import numpy as np
import os
from typing import Optional

from ..SubNets.resnet.resnet_utils import myResnet
from ..SubNets.resnet import resnet
from .SHARK import TextEncoder, GateModule, Fushion

class MultiModalBartConfig(BartConfig):
    def __init__(
            self,
            activation_dropout=0.0,
            extra_pos_embeddings=2,
            activation_function="gelu",
            vocab_size=50320,
            image_feature_size=2048 + 4,
            d_model=1024,
            encoder_ffn_dim=4096,
            encoder_layers=12,
            encoder_attention_heads=16,
            decoder_ffn_dim=4096,
            decoder_layers=12,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            attention_dropout=0.0,
            dropout=0.1,
            max_position_embeddings=1024,
            init_std=0.02,
            classif_dropout=0.0,
            num_labels=1,
            num_attributes=1,
            num_relations=1,
            is_encoder_decoder=True,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            img_feat_id=50273,
            cls_token_id=50276,
            normalize_before=False,
            add_final_layer_norm=False,
            scale_embedding=False,
            normalize_embedding=True,
            static_position_embeddings=False,
            add_bias_logits=False,
            decoder_start_token_id=0,
            partial_load=(),
            lm_loss_factor=1.0,
            mrm_loss_factor=1.0,
            attribute_loss_factor=1.0,
            relation_loss_factor=1.0,
            **common_kwargs
    ):
        super(MultiModalBartConfig, self).__init__(
            activation_dropout=activation_dropout,
            extra_pos_embeddings=extra_pos_embeddings,
            activation_function=activation_function,
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            attention_dropout=attention_dropout,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            init_std=init_std,
            classif_dropout=classif_dropout,
            num_labels=num_labels,
            is_encoder_decoder=is_encoder_decoder,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            normalize_before=normalize_before,
            add_final_layer_norm=add_final_layer_norm,
            scale_embedding=scale_embedding,
            normalize_embedding=normalize_embedding,
            static_position_embeddings=static_position_embeddings,
            add_bias_logits=add_bias_logits,
            decoder_start_token_id=decoder_start_token_id,
            **common_kwargs
        )

        self.image_feature_size = image_feature_size
        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        self.partial_load = partial_load
        self.num_attributes = num_attributes
        self.num_relations = num_relations
        self.lm_loss_factor = lm_loss_factor
        self.mrm_loss_factor = mrm_loss_factor
        self.attribute_loss_factor = attribute_loss_factor
        self.relation_loss_factor = relation_loss_factor

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
    
class MultimodalBartEncoder(nn.Module):
    def __init__(self, args, tokenizer):
        super(MultimodalBartEncoder, self).__init__()
        bartModel = BartModel.from_pretrained('facebook/bart-base')
        bartModel.resize_token_embeddings(len(tokenizer))
        self.encoder = bartModel.encoder

        self.embed_tokens = self.encoder.embed_tokens
        self.embed_images = ImageEmbedding(2048, self.embed_tokens.embedding_dim)
        self.embed_positions = self.encoder.embed_positions

        self.img_feat_id = tokenizer.tokenize('[IFEAT]')
        self.img_begin_id = tokenizer.tokenize('[BIMG]')
        self.img_end_id = tokenizer.tokenize('[EIMG]')

    def _embed_multi_modal(self, text_feats, video_feats, video_ids):
        """
        :param text_feats: Tensor, [[input_ids, input_mask, segment_ids, noun_mask]] 
        :param video_feats: Tensor, [batch_size, 49, 2048]
        :param video_ids: Tensor, [[input_ids, input_attention_mask]]
        :return:
            input_ids: sequence of both text and visual ids
            input_masks: sequence of both text and visula masks
            token_embed: concate text and visual into one embedding
        """
        input_ids = torch.cat([video_ids[:,0], text_feats[:,0]], dim=1)
        # print(input_ids, input_ids.shape)
        # mask = (input_ids == self.img_feat_id) | (input_ids == self.img_begin_id) | (
        #     input_ids == self.img_end_id)
        
        # print(input_ids, input_ids.shape, mask)
        
        token_embed = self.embed_tokens(input_ids)
        image_embed = self.embed_images(video_feats)

        if not image_embed[0].dtype == torch.float32:
            token_embed = token_embed.half()
        # print(token_embed, token_embed.shape)
        # print(image_embed, type(image_embed))
        # print(len(image_embed))
        # token_embed = torch.cat([image_embed, token_embed], dim=1)
        # for index, value in enumerate(image_embed):
        #     if len(value) > 0:
        #         token_embed[index, mask[index]] = value

        for index, value in enumerate(image_embed):
            for i in range(len(value)):
                token_embed[index, i] = value[i]
        # print(video_feats[:,1].shape, text_feats[:,1].shape)
        input_masks = torch.cat([video_ids[:,1], text_feats[:,1]], dim=1)
        
        padding = torch.zeros([len(input_ids), len(input_ids[0]) - len(text_feats[:,3][0])])
        # print(padding, padding.shape)
        noun_masks = torch.cat([padding, text_feats[:,3]], dim=1)
        # for i in range(len(input_ids)):
        #     padding = [0] * (len(input_ids[i]) - len(text_feats[:,3][i]))
        #     print(padding)
        #     noun_masks = padding + text_feats[:,3].tolist()
        #     noun_masks = torch.tensor(noun_masks)

        # print(input_ids, input_masks, token_embed, noun_masks)
        # print(input_ids.shape, input_masks.shape, token_embed.shape, noun_masks.shape)
        return input_ids, input_masks, token_embed, noun_masks
    
    def forward(self,
                text_feats,
                video_feats,
                video_ids,
                output_attentions=False,
                output_hidden_states=True,
                training=True):
        
        input_ids, input_masks, input_embeds, noun_masks = self._embed_multi_modal(text_feats, video_feats, video_ids)
        input_ids = input_ids * self.encoder.embed_scale
        input_masks = input_masks * self.encoder.embed_scale
        input_embeds = input_embeds * self.encoder.embed_scale
        noun_masks = noun_masks * self.encoder.embed_scale
        
        if input_masks is not None:
            attention_mask = _expand_mask(input_masks, input_embeds.dtype)

        embed_pos = self.embed_positions(input_ids)

        x = input_embeds + embed_pos
        x = self.encoder.layernorm_embedding(x)
        x = F.dropout(x, self.encoder.dropout, training)
        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.encoder.layers:
            if output_hidden_states:
                encoder_states.append(x)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.encoder.layerdrop):  # skip the layer
                attn = None
            else:
                # print(x.shape, attention_mask.shape, attention_mask)
                outputs = encoder_layer(x, attention_mask, layer_head_mask=None, output_attentions=output_attentions)
                # print(outputs)
                x = outputs[0]
                # attn = outputs[1]

            if output_attentions:
                all_attentions.append(attn)

        # if self.encoder.layer_norm:
        #     x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        # encoder_states = [
        #     hidden_state.transpose(0, 1) for hidden_state in encoder_states
        # ]
        # x = x.transpose(0, 1)

        return {
            'last_hidden_state': x,
            'hidden_states': encoder_states,
            'attentions': all_attentions,
            'noun_masks': noun_masks,
            'attention_masks': input_masks
        }

class A3M(BartPreTrainedModel):
    def __init__(self, config: MultiModalBartConfig, args, tokenizer):
        super(A3M, self).__init__(config)
        self.config = config
                
        net = getattr(resnet, 'resnet152')()
        net.load_state_dict(torch.load('/Users/admin/Documents/Projects/Multi-Intent-Recognition/a3m/backbones/FusionNets/resnet152-b121ed2d.pth'))
        self.img_encoder = myResnet(net, True, args.device)
        self.img_encoder.to(args.device)

        self.multimodal_encoder = MultimodalBartEncoder(args, tokenizer)

        self.noun_linear = nn.Linear(768,768)
        self.multi_linear = nn.Linear(768,768)
        self.att_linear = nn.Linear(768*2,1)
        self.linear=nn.Linear(768*2,1)
        self.alpha_linear1=nn.Linear(768,768)
        self.alpha_linear2=nn.Linear(768,768)

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
        noun_position=torch.tensor(noun_position).to(self.device)
        noun_embed=torch.zeros(feature.shape[0],max_noun_num,feature.shape[-1]).to(self.device)
        for i in range(len(feature)):
            if noun_position[i].dtype != torch.int32 and noun_position[i].dtype != torch.int64:
                print(noun_position[i], noun_num[i])
            else:
                noun_embed[i]=torch.index_select(feature[i],dim=0,index=noun_position[i])
            # print(noun_embed)
            noun_embed[i,noun_num[i]:]=torch.zeros(max_noun_num-noun_num[i],feature.shape[-1])
            # print(noun_embed)
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
        # print(alpha.shape, encoder_outputs.shape, att_features.shape)
        alpha = alpha.repeat(1, 1, 768)

        encoder_outputs = torch.mul(1-alpha, encoder_outputs) + torch.mul(alpha, att_features)

        return encoder_outputs
    
    def forward(self, text_feats, video_feats, video_ids, training):
        imgs_f, img_mean, img_att = self.img_encoder(video_feats)
        img_att = img_att.view(-1, 2048, 49).permute(0, 2, 1)
        
        encoder_outputs = self.multimodal_encoder(text_feats, img_att, video_ids, training=training)
        encoder_output = encoder_outputs['last_hidden_state']
        hidden_states = encoder_outputs['hidden_states']
        encoder_masks = encoder_outputs['attention_masks']
        src_embed_outputs = hidden_states[0]

        noun_embed = self.get_noun_embed(encoder_output, encoder_outputs['noun_masks'])
        visual_feats = self.noun_attention(encoder_output, noun_embed)

        return visual_feats

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class OurModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(OurModel, self).__init__()
        # tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        # additional_special_tokens = ['[CLS]', '[SEP]', '[BIMG]', '[EIMG]', '[IFEAT]']
        # tokenizer.add_tokens(additional_special_tokens) 
        # unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        # tokenizer.unique_no_split_tokens = unique_no_split_tokens + additional_special_tokens

        self.config = MultiModalBartConfig.from_pretrained('facebook/bart-base')
        self.a3m = A3M(self.config, args, tokenizer)
        self.text_encoder = TextEncoder(args, self.a3m.multimodal_encoder.embed_tokens)
        self.gate = GateModule(args)
        self.fushion = Fushion(args)
        
        self.pooler = Pooler(self.config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)

    def forward(self, text_feats, video_feats, audio_feats, video_ids, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats, training=True):
        text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb = \
            self.text_encoder(text_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats)

        new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt = self.gate(text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb)

        text_new_feats = self.fushion(text_emb, new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt)

        video_new_feats = self.a3m(text_feats, video_feats, video_ids, training=training)

        # print(text_new_feats.shape, video_new_feats.shape)

        features = torch.cat([text_new_feats, video_new_feats], dim=1)

        # print(features.shape, self.config.hidden_size)

        output = self.dropout(self.pooler(features))
        # print(output.shape)
        logits = self.classifier(output)
        # print(logits.shape)
        return logits

# def invert_mask(attention_mask):
#     """Turns 1->0, 0->1, False->True, True-> False"""
#     assert attention_mask.dim() == 2
#     return attention_mask.eq(0)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
