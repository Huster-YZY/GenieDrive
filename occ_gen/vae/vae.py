import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, v, k, q):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Scaled Dot-Product Attention
        attention_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)

        attention_weights = F.softmax(attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(output)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff = None, dropout_rate = 0.5):
        super(TransformerBlock, self).__init__()
        if d_ff is None:
            d_ff = d_model

        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)

        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout2(ffn_output))


class CNN_encoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CNN_encoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=dim_in, out_channels=dim_out, kernel_size=5, stride=2, padding=2)  
        self.conv2 = nn.Conv3d(in_channels=dim_out, out_channels=dim_out, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
    
class CNN_decoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CNN_decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels=dim_in, out_channels=dim_out, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=dim_out, out_channels=dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x
    
class OccupancyVAE(nn.Module):
    def __init__(self, occ_shape, num_classes, expansion=8, d_model=16, num_heads=2, depth=2, d_ff=None, dropout_rate=0.5):
        super(OccupancyVAE, self).__init__()
        if d_ff is None:
            d_ff = d_model
        self.num_classes = num_classes
        
        self.class_embeds = nn.Embedding(num_classes, expansion)

        X, Y, Z = occ_shape
        downsample_rate = 4
        shape_dict = dict(x=X//downsample_rate, y=Y//downsample_rate, z=Z//downsample_rate)
        plane_names = ['xy', 'xz', 'yz']
        reduce_dim = {'xy': 'z', 'xz': 'y', 'yz': 'x'}
        self.reduce_dim = reduce_dim
        self.shape_dict = shape_dict

        self.tokens = nn.ParameterDict()
        self.pos_encodings = nn.ParameterDict()
        self.transformers = nn.ModuleDict()
        self.rearrange1 = dict()
        self.rearrange2 = dict()

        for plane in plane_names:
            self.tokens[plane] = nn.Parameter(torch.randn(1, 1, d_model))
            self.pos_encodings[plane] = nn.Parameter(torch.randn(1, shape_dict[reduce_dim[plane]] + 1, d_model))
            # self.transformers[plane] = nn.Sequential(*[TransformerBlock(d_model*2, num_heads, d_ff, dropout_rate) for _ in range(depth)])
            self.transformers[plane] = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(depth)])
            self.rearrange1[plane] = (f'b c x y z -> 'f'(b {" ".join(plane)}) {reduce_dim[plane]} c')  # bcxyz->(bxy)zc
            self.rearrange2[plane] = (f'(b {" ".join(plane)}) c -> b c {" ".join(plane)}', {key: shape_dict[key] for key in plane})
            

        self.conv = nn.Conv3d(expansion, d_model, kernel_size=1)
        # self.vox_encoder = CNN_encoder(dim=d_model) #channel mult*2
        # self.vox_decoder = CNN_decoder(dim=d_model)
        self.vox_encoder = CNN_encoder(dim_in=d_model, dim_out = d_model)

        
        self.vox_decoder = CNN_decoder(dim_in=d_model, dim_out = d_model)

        #vae
        self.fc_mu_logvar = nn.ModuleList([nn.Linear(d_model, d_model * 2) for _ in range(3)])

        #occ decoder
        self.decode_pos_embed = nn.Parameter(torch.randn(1, 1, X//downsample_rate, Y//downsample_rate, Z//downsample_rate))
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, self.num_classes),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def encode(self, x, concat = False):
        # import pdb; pdb.set_trace()
        x = self.class_embeds(x)
        x = rearrange(x, 'b x y z c-> b c x y z').contiguous()
        x = self.conv(x)
        x = self.vox_encoder(x)

        xy = self.forward_plane(x, 'xy')
        xz = self.forward_plane(x, 'xz')
        yz = self.forward_plane(x, 'yz')

        #add vae sample
        mus, logvars = list(), list()
        for i, plane in enumerate([xy, xz, yz]):
            _, _, dim1, dim2 = plane.shape
            plane = rearrange(plane, 'b c d1 d2 -> b (d1 d2) c')
            mu_logvar = self.fc_mu_logvar[i](plane)
            mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
            mus.append(rearrange(mu, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))
            logvars.append(rearrange(logvar, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))

        xy, xz, yz = [self.latent_sample(mus[i], logvars[i]) for i in range(3)]
        # xy, xz, yz = [mus[i] for i in range(3)]
        
        if concat:
            return torch.cat([xy, xz, yz], dim=-1)
        return xy, xz, yz, mus, logvars
    
    def encode_wo_sample(self, x):
        x = self.class_embeds(x)
        x = rearrange(x, 'b x y z c-> b c x y z').contiguous()
        x = self.conv(x)
        x = self.vox_encoder(x)

        xy = self.forward_plane(x, 'xy')
        xz = self.forward_plane(x, 'xz')
        yz = self.forward_plane(x, 'yz')
        return torch.cat([xy, xz, yz], dim=-1)
    
    def post_sample(self, x):
        xy, xz, yz = torch.split(x, [self.shape_dict['y'], self.shape_dict['z'], self.shape_dict['z']], dim=-1)
        mus, logvars = list(), list()
        for i, plane in enumerate([xy, xz, yz]):
            _, _, dim1, dim2 = plane.shape
            plane = rearrange(plane, 'b c d1 d2 -> b (d1 d2) c')
            mu_logvar = self.fc_mu_logvar[i](plane)
            mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
            mus.append(rearrange(mu, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))
            logvars.append(rearrange(logvar, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))
        xy, xz, yz = [self.latent_sample(mus[i], logvars[i]) for i in range(3)]
        return torch.cat([xy, xz, yz], dim=-1), mus, logvars
    
    def generate(self, latent):
        xy, xz, yz = torch.split(latent, [self.shape_dict['y'], self.shape_dict['z'], self.shape_dict['z']], dim=-1)
        x = self.decode(xy, xz, yz)
        # x = torch.argmax(x, dim=1)
        return x
    
    def decode_occ(self, latent):
        probs = self.generate(latent)
        return torch.argmax(probs, dim=1)
    
    def decode(self, xy, xz, yz):
        xy = repeat(xy, 'b c x y -> b c x y z', z=self.shape_dict['z'])
        xz = repeat(xz, 'b c x z -> b c x y z', y=self.shape_dict['y'])
        yz = repeat(yz, 'b c y z -> b c x y z', x=self.shape_dict['x'])

        vox_ft = xy * xz * yz  + self.decode_pos_embed
        # vox_ft = xy * xz * yz

        vox_ft = self.vox_decoder(vox_ft)
        vox_ft = rearrange(vox_ft, 'b c x y z -> b x y z c')
        x = self.cls_head(vox_ft)
        x = rearrange(x, 'b x y z c -> b c x y z')
        return x
    
    def forward(self, x):
        xy, xz, yz, mus, logvars= self.encode(x)
        return self.decode(xy, xz, yz), mus, logvars
    
    def forward_plane(self, x, plane_name):
        x = rearrange(x, self.rearrange1[plane_name])
        token = repeat(self.tokens[plane_name], '1 1 c -> b 1 c', b=x.shape[0])
        plane = torch.cat([x, token], dim=1)
        plane = plane + self.pos_encodings[plane_name]
        plane = self.transformers[plane_name](plane)[:, 0]  # B*X*Y, 1, C
        plane = rearrange(plane, self.rearrange2[plane_name][0], **self.rearrange2[plane_name][1]) # B, X, Y, C
        return plane
    
    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def encode_decode(self, x):
        x, _, _ = self.forward(x)
        x = torch.argmax(x, dim=1)
        return x.squeeze(0)