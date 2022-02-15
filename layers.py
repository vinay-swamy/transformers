#%%
import torch 
from torch import nn 
import numpy as np
import torch.nn.functional as F
#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, input_output_dim, head_dim,  num_heads):
        super(MultiHeadAttention, self).__init__()
        assert input_output_dim % num_heads == 0, "d_model % num_heads should be zero."
        self.input_output_dim = input_output_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        # In single headed attention, the output projection is generally smaller than the 
        # input dimension. However, in the multi head case, we are going to have the same 
        # input & ouptut dimension for the projection, where portions of the output dim
        # correspond to the different heads. So we have the same *amount* of data 
        # but have more, smaller matrices 
        self.query_proj = nn.Linear(self.input_output_dim, self.head_dim * num_heads, bias = False)
        self.key_proj = nn.Linear(self.input_output_dim, self.head_dim * num_heads, bias = False)
        self.value_proj = nn.Linear(self.input_output_dim, self.head_dim* num_heads, bias = False)

        ## project the attention output back to the same dimension as input
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.input_output_dim, bias = False)

    def forward(
            self,
            query,
            key,
            value,
            mask=None): 
        batchsize = query.shape[0]
        seqlen = query.shape[1]
        
        ## reshape  output of query such that split out the different head matrices, and then reorder them so 
        ## that we have batchsize*numheads, seqlen, head dim. we are essentially treating  the seperate head matrices as different batches

        query = self.query_proj(query)  
        query = query.view(batchsize, seqlen, self.num_heads, self.head_dim) 
        query = query.permute(2, 0, 1, 3).contiguous().view(batchsize*self.num_heads, seqlen, self.head_dim)

        key = self.key_proj(key)  
        key = key.view(batchsize, seqlen, self.num_heads, self.head_dim) 
        key = key.permute(2, 0, 1, 3).contiguous().view(batchsize*self.num_heads, seqlen, self.head_dim)

        value = self.value_proj(value)  
        value = value.view(batchsize, seqlen, self.num_heads, self.head_dim) 
        value = value.permute(2, 0, 1, 3).contiguous().view(batchsize*self.num_heads, seqlen, self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        attn_values, attn_scores = self.scaled_dotproduct_attn(query, key, value, mask)

        attn_values = attn_values.view(self.num_heads, batchsize, seqlen, self.head_dim)
        attn_values = attn_values.permute(1, 2, 0, 3).contiguous().view(batchsize, seqlen, self.num_heads * self.head_dim)  
        attn_values = self.out_proj(attn_values)
        return attn_values, attn_scores

    def scaled_dotproduct_attn(self, query, key, value, mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class PositionalEncoder(nn.Module):
    def __init__(self, seqlen, model_dim):
        ## generate positional encoding for batch-first tensors (batch, seqlen, model_dim)
        ## based vaswani et al paper 
        super(PositionalEncoder, self).__init__()
        seq_vec = torch.arange(seqlen).unsqueeze(1)
        dim_vec = torch.arange(model_dim)
        div_vec = torch.pow(10000, dim_vec/model_dim)
        pe = torch.zeros((1, seqlen, model_dim))
        pe[0,0::2,:] = torch.sin(seq_vec[0::2]  / div_vec)
        pe[0,1::2, :] = torch.cos(seq_vec[1::2] /div_vec)
        self.register_buffer('positional_encoding', pe) 
    def forward(self, x):
        x=x + self.positional_encoding
        return x 


class DefaultTransfomerBlock(nn.Module):
    """
    Implementing the tranformer block as close to what was described in the vaswani paper
    Not as flexible as the standard pytorch implementation, but definitely easier to follow

    """
    def __init__(self,input_output_dim, head_dim, num_heads, ff_internal_dim, dropout_frac):
        super(DefaultTransfomerBlock, self).__init__()
        self.mha = nn.Sequential(
            MultiHeadAttention(input_output_dim, head_dim, num_heads),
            nn.Dropout(dropout_frac)
        )
        self.ln1 = nn.LayerNorm(input_output_dim) ## use default eps, but how does changing eps change performance 

        self.ff = nn.Sequential(
            nn.Linear( input_output_dim, ff_internal_dim, bias = True ) ,
            nn.ReLU(),
            nn.Linear(ff_internal_dim, input_output_dim, bias = True),
            nn.Dropout(dropout_frac)
            )
        self.ln2 = nn.LayerNorm(input_output_dim)

    def forward(self, x):
        x = self.mha(x, x, x) + x
        x = self.ln1(x)
        x = self.ff(x) + x 
        x = self.ln2(x) 
        return x

class DecoderTranformerBlock(nn.module):
    def __init__(self, input_output_dim, head_dim, num_heads):
        super(DecoderTranformerBlock, self).__init__()
        self.mha = MultiHeadAttention(input_output_dim, head_dim, num_heads)
        self.transformer_block = DefaultTransfomerBlock()
    



# %%


