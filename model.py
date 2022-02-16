import torch 
from torch import nn 
import layers as L

class MSA2PWMtransformer(nn.Module):
    """
    Basic architecture 
    Use internal AlphaFold "Single Representation" as the input 
    This has shape (N_res, N_channel), Assuming that N_channel = 20, Strat right now is to use use a 
    linear layer(N_chanel, io_dim) to upscale to the io dim
    This feeds through the standard transformer encoder-decoder stack. The final 

    TODO:
        add masks
    """
    
    def __init__(self, input_output_dim,head_dim, num_heads,ff_internal_dim, n_input_channel, pwm_dim, dropout_frac ):
        super(MSA2PWMtransformer, self).__init__()
        
        self.channel_embedding = nn.Linear(n_input_channel,input_output_dim )
        self.encoder_pe = L.PositionalEncoder(input_output_dim)## fix this to a function to account for variable size inputs 
        self.encoder_tfmr1 = L.DefaultTransfomerBlock(input_output_dim, 
                                                      head_dim, num_heads, 
                                                      ff_internal_dim, 
                                                      dropout_frac)
        self.encoder_trmr2 = L.DefaultTransfomerBlock(input_output_dim, 
                                                      head_dim, num_heads, 
                                                      ff_internal_dim, 
                                                      dropout_frac)
        self.pwm_embedding = nn.Linear(pwm_dim, input_output_dim)
        self.decoder_pe = L.PositionalEncoder(input_output_dim)
        self.decoder_tfmr1 = L.DecoderTranformerBlock(input_output_dim, 
                                                      head_dim, num_heads, 
                                                      ff_internal_dim, 
                                                      dropout_frac)
        self.decoder_tfmr2 = L.DecoderTranformerBlock(input_output_dim, 
                                                      head_dim, num_heads, 
                                                      ff_internal_dim, 
                                                      dropout_frac)
        self.final_linear = nn.Linear(input_output_dim, pwm_dim)
        self.final_sm = nn.Softmax()
    def forward(self, x_enc, x_dec ):
        ## encoder part
        x_enc = self.channel_embedding(x_enc)
        x_enc = self.encoder_pe(x_enc)
        x_enc = self.encoder_tfmr1(x_enc)
        x_enc = self.encoder_tfmr2(x_enc)
       ## decoder part
        x_dec = self.pwm_embedding(x_dec)
        x_dec = self.decoder_pe(x_dec)
        x_dec = self.decoder_tfmr1(x_dec, x_enc)
        x_dec = self.decoder_tfmr2(x_dec, x_enc)
        logits = self.final_linear(x_dec)
        return self.final_sm(logits)

                                                    

