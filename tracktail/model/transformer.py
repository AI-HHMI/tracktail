import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange
# from xformers.ops import memory_efficient_attention


class TimeSpaceTransformer(nn.Module): 

    def __init__(self, input_dim, embedding_dim, 
                 output_dim, n_time_space_blocks = 6, 
                 n_heads = 8, n_virtual = 64, 
                 embedding_factor = 4, vc_head = False, 
                 device = None, **activation_kwargs): 
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.n_time_space_blocks = n_time_space_blocks
        self.n_heads = n_heads
        self.embedding_factor = embedding_factor
        self.vc_head = vc_head
        self.activation_kwargs = activation_kwargs

        # initialize virtual tracks
        self.n_virtual = n_virtual
        self.virtual_tracks = nn.Parameter(torch.randn((1, 1, 
            self.n_virtual, self.embedding_dim), device = device))

        # linear layers
        self.embedding = nn.Linear(self.input_dim, self.embedding_dim)

        if self.vc_head: 
            self.mlp_head = nn.Linear(self.embedding_dim, self.output_dim - 2)
            self.vc_head = nn.Linear(self.embedding_dim, 2)

        else: 
            self.mlp_head = nn.Linear(self.embedding_dim, self.output_dim)

        # initialize attention blocks
        self.time_blocks = self.init_attn_blocks(cross_attn = False)
        self.space_blocks_vxv = self.init_attn_blocks(cross_attn = False)
        self.space_blocks_pxv = self.init_attn_blocks(cross_attn = True)
        self.space_blocks_vxp = self.init_attn_blocks(cross_attn = True)

        # initialize weights for linear layer 
        self.apply(self.init_weights)

    def init_attn_blocks(self, cross_attn = False):

        modules = []

        for i in range(self.n_time_space_blocks):

            block = AttentionBlock(
                embedding_dim = self.embedding_dim, 
                context_dim = self.embedding_dim,
                embedding_factor = self.embedding_factor, 
                n_heads = self.n_heads, 
                cross_attn = cross_attn, 
                **self.activation_kwargs
            )
            
            modules.append(block)

        return nn.ModuleList(modules)

    def init_weights(self, module): 

        if isinstance(module, nn.Linear): 
            nn.init.xavier_normal_(module.weight)
            # nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: 
                nn.init.constant_(module.bias, 0)

        torch.nn.init.trunc_normal_(self.mlp_head.weight, std = 0.001)
        if self.vc_head: 
            torch.nn.init.trunc_normal_(self.vc_head.weight, std = 0.001)
 
    def forward(self, x): 

        z = self.embedding(x)
        B, S, n, H = z.shape 

        z_virtual = self.virtual_tracks.repeat((B, S, 1, 1))
        z = torch.cat([z, z_virtual], dim = 2)
        _, _, N, _ = z.shape 

        for i in range(self.n_time_space_blocks): 

            z_time = rearrange(z, 'b s n h -> (b n) s h')
            z_time = self.time_blocks[i](z_time)

            z_space = rearrange(z_time, '(b n) s h -> (b s) n h', b = B, n = N)
            z_space_p = z_space[:, :n, :]
            z_space_v = z_space[:, n:, :]

            z_space_v = self.space_blocks_vxp[i](z_space_v, z_space_p)
            z_space_v = self.space_blocks_vxv[i](z_space_v)
            z_space_p = self.space_blocks_pxv[i](z_space_p, z_space_v)

            z = torch.cat([z_space_p, z_space_v], dim = 1)
            z = rearrange(z, '(b s) n h -> b s n h', b = B, s = S)

        z = z[:, :, :n, :]

        outputs = self.mlp_head(z)

        if self.vc_head: 
            vc_outputs = self.vc_head(z)
            outputs = torch.cat((outputs, vc_outputs), dim = -1)

        return outputs 


class Attention(nn.Module): 

    def __init__(self, 
                query_dim, 
                embedding_dim, 
                context_dim = None, 
                n_heads = 10):

        super().__init__()

        self.query_dim = query_dim
        self.embedding_dim = embedding_dim

        if context_dim is None: 
            self.context_dim = query_dim
        else: 
            self.context_dim = context_dim

        self.n_heads = n_heads
        self.head_dim = self.embedding_dim // self.n_heads

        self.q_transf = nn.Linear(self.query_dim, self.embedding_dim)
        self.k_transf = nn.Linear(self.context_dim, self.embedding_dim)
        self.v_transf = nn.Linear(self.context_dim, self.embedding_dim)
        self.out_transf = nn.Linear(self.embedding_dim, self.query_dim)

    def reshape_attn(self, params, b, n):

        params = rearrange(
            params, 
            'b n (n_heads head_dim) -> b n_heads n head_dim',
            head_dim = self.head_dim, 
            n_heads = self.n_heads
        )

        return params

    def forward(self, x, context = None, bias = None): 

        if context is None: 
            context = x

        B, N, H = x.shape
        B2, N2, H2 = context.shape

        qs = self.reshape_attn(self.q_transf(x), B, N)
        ks = self.reshape_attn(self.k_transf(context), B2, N2)
        vs = self.reshape_attn(self.v_transf(context), B2, N2)

        # out = memory_efficient_attention(
        #     query = qs,
        #     key = ks,
        #     value = vs,
        #     p = 0.0)

        similarities = torch.div(
            torch.matmul(qs, ks.transpose(-2, -1)),
            torch.sqrt(torch.tensor(self.head_dim, device = x.device))
        )

        attn_weights = torch.softmax(similarities, dim = -1) 

        out = torch.matmul(attn_weights, vs)

        out = rearrange(out, 'b n_heads n head_dim -> b n (n_heads head_dim)', 
                        n_heads = self.n_heads, head_dim = self.head_dim)

        out = self.out_transf(out) 

        return out


class AttentionBlock(nn.Module): 

    def __init__(self, 
                 embedding_dim, 
                 context_dim = None, 
                 n_heads = 8, 
                 cross_attn = False, 
                 embedding_factor = 4, 
                 **activation_kwargs):

        super().__init__()

        self.cross_attn = cross_attn

        self.layernorm1 = nn.LayerNorm(
            normalized_shape = embedding_dim, 
            eps = 1e-06, 
            elementwise_affine = False
        )

        if context_dim is None: 
            context_dim = embedding_dim
        
        if self.cross_attn:
            self.context_layernorm1 = nn.LayerNorm(embedding_dim)

        self.attention = Attention(
            query_dim = embedding_dim,
            embedding_dim = embedding_dim, 
            context_dim = context_dim, 
            n_heads = n_heads
        )

        self.layernorm2 = nn.LayerNorm(
            normalized_shape = embedding_dim, 
            eps = 1e-06, 
            elementwise_affine = False
        )

        self.mlp = MLP(
            input_dim = embedding_dim, 
            embedding_dim = int(embedding_dim * embedding_factor), 
            output_dim = embedding_dim,
            **activation_kwargs
        )

    def forward(self, x, context = None):

        if self.cross_attn:
            x = x + self.attention(self.layernorm1(x), context = self.context_layernorm1(context))
        else: 
            x = x + self.attention(self.layernorm1(x))

        x = x + self.mlp(self.layernorm2(x))

        return x 


class MLP(nn.Module): 

    def __init__(self, 
                input_dim, 
                embedding_dim,
                output_dim,  
                activation = nn.GELU,
                **activation_kwargs):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.GELU(**activation_kwargs)
        self.fc2 = nn.Linear(embedding_dim, output_dim) 

    def forward(self, x): 

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x 
