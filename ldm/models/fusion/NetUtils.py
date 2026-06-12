# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from einops import rearrange

# class LayerNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = self.norm(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
#         return x

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         qkv = self.qkv(x)
#         q, k, v = qkv.chunk(3, dim=1)

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = F.normalize(q, dim=-1)
#         k = F.normalize(k, dim=-1)
#         attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=H, w=W)
#         return self.proj(out)

# class FeedForward(nn.Module):
#     def __init__(self, dim, expansion=2):
#         super().__init__()
#         hidden = int(dim * expansion)
#         self.block = nn.Sequential(
#             nn.Conv2d(dim, hidden, 1),
#             nn.GELU(),
#             nn.Conv2d(hidden, dim, 1)
#         )

#     def forward(self, x):
#         return self.block(x)

# class RestormerBlock(nn.Module):
#     def __init__(self, dim, num_heads=4):
#         super().__init__()
#         self.norm1 = LayerNorm(dim)
#         self.attn = Attention(dim, num_heads)
#         self.norm2 = LayerNorm(dim)
#         self.ffn = FeedForward(dim)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#         return x

# class RestormerFusionLayer(nn.Module):
#     def __init__(self, in_channels=6, embed_dim=48, out_channels=3):
#         super().__init__()
#         self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
#         self.blocks = nn.Sequential(
#             RestormerBlock(embed_dim),
#             RestormerBlock(embed_dim)
#         )
#         self.output = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = self.patch_embed(x)
#         x = self.blocks(x)
#         x = self.output(x)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out
    
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
    
class RestormerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm(channels)  # spatial LayerNorm
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm(channels)
        self.ffn = GDFN(channels, expansion)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.ffn(x)
        return x + res

class RestormerFusionLayer(nn.Module):
    def __init__(self, in_channels=6, embed_channels=48, out_channels=3, num_blocks=4, num_heads=4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, embed_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[
            RestormerBlock(embed_channels, num_heads=num_heads) for _ in range(num_blocks)
        ])
        self.proj = nn.Conv2d(embed_channels, out_channels, kernel_size=3, padding=1)
        self.compress = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, ir, vis):
        feature = torch.cat((ir,vis),1) # (B, 6, H, W)
        x = self.embed(feature)  # (B, embed, H, W)
        x = self.blocks(x)
        x = self.proj(x)  # (B, 6, H, W)
        out = x + vis
        out = self.compress(self.relu(out))  # (B, 3, H, W)
        return out
