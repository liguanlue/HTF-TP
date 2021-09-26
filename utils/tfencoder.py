import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.t_nlayers)])

    def forward(self, enc_outputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_outputs [batch_size, src_len, d_model]

        enc_outputs: [batch_size, src_len, d_model]
        enc_self_attn: [batch_size, n_heads, src_len, src_len]
        '''

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            # enc_self_attns.append(enc_self_attn)
        # return enc_outputs, enc_self_attns
        return enc_outputs

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs):
        '''
        # enc_inputs: [batch_size, src_len, d_model]  enc_inputs to same Q,K,V
        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.t_nhead = args.t_n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_model = args.d_model

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.t_n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.t_n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.t_n_heads, bias=False)
        self.fc = nn.Linear(self.t_n_heads * self.d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]

        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.t_n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.t_n_heads, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.t_n_heads, self.d_v).transpose(1,2)

        if(attn_mask!=None):
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.t_n_heads, 1,1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
            # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
            context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        else:
            context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.t_n_heads * self.d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        # return nn.LayerNorm(self.d_model).cuda()(output + residual), attn
        return nn.LayerNorm(self.d_model)(output + residual), attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # scores : [batch_size, n_heads, len_q, len_k]
        if(attn_mask!=None):
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = args.d_model
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model)(output + residual)  # [batch_size, seq_len, d_model]
        # return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]

