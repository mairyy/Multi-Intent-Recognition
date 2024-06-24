# test new flow => eval_f1 = 0.0114
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
import torch.nn.functional as F
from torch import nn
import torch
from ..SubNets.AlignNets import AlignSubNet

class TextEncoder(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_embedding(
        self, 
        text,
        head_mask=None,
        inputs_embeds=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]
        
        embedddings = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)
        
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
        #     attention_mask, input_ids.size()
        # )

        # outputs = self.encoder(
        #     embedddings,
        #     attention_mask=extended_attention_mask,
        #     output_attentions=self.config.output_attentions,
        #     output_hidden_states=self.config.output_hidden_states
        # )
        
        # output = outputs.last_hidden_state #torch.size([batch_size, n_Tokens, text_feat_dim])

        return (embedddings, 
                attention_mask, 
                head_mask,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                output_attentions,
                output_hidden_states,
            )
    
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

        return text_emb, xReact_comet_emb[0], xWant_comet_emb[0], xReact_sbert_emb[0], xWant_sbert_emb[0]
    
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
    
class SharkModule(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.text_encoder = TextEncoder(self.config, args)
        self.gate = GateModule(args)
        self.fushion = Fushion(args)
        self.mag = MAG(self.config, args)
        self.mm_encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)

    def forward(self, text_feats, video_feats, audio_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats):
        text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb = \
            self.text_encoder(text_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats)

        new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt = self.gate(text_emb[0], xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb)

        text_feats = self.fushion(text_emb[0], new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt)
        # text_feats = self.fushion(text_emb, xReact_comet_emb, xWant_comet_emb) #remove sbert
        # text_feats = self.fushion(text_emb, xReact_sbert_emb, xWant_sbert_emb) #remove comet

        # output = torch.mean(self.mag(text_feats, video_feats, audio_feats), dim=1) #full model
        # output = torch.mean(text_feats, dim=1) #remove audio&video

        mag_outputs = self.mag(text_feats, video_feats, audio_feats)

        encoder_outputs = self.mm_encoder(
            mag_outputs,
            head_mask=text_emb[2],
            encoder_hidden_states=text_emb[3],
            encoder_attention_mask=text_emb[4],
            output_attentions=text_emb[5],
            output_hidden_states=text_emb[6],
        )

        sequence_output = encoder_outputs[0]
        # sequence_output = torch.mean(encoder_outputs.last_hidden_state, dim=1)
        pooler_output = self.pooler(sequence_output)
        output = self.dropout(pooler_output)
        logits = self.classifier(output)
        # print(output.shape, logits.shape)
        return logits
    
class Shark(nn.Module):
    def __init__(self, args):
        super(Shark, self).__init__()
        self.model = SharkModule.from_pretrained(args.text_backbone, cache_dir = args.cache_path, args = args)

    def forward(self, text_feats, video_feats, audio_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats):
        logits = self.model(
            text_feats, 
            video_feats, 
            audio_feats, 
            xReact_comet_feats, 
            xWant_comet_feats, 
            xReact_sbert_feats, 
            xWant_sbert_feats
        )
        return logits