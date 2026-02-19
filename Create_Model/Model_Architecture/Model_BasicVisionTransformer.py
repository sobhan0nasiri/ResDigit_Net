import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        
        x = self.proj(x) 
        
        x = x.flatten(2) 
        
        x = x.transpose(1, 2)
        
        return x

class ViTPositionalAndCls(nn.Module):
    
    
    def __init__(self, n_patches=196, embed_dim=768, dropout_rate=0.1):
        super().__init__()
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
    
        self.pos_drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
    
        batch_size = x.shape[0]
    
        cls_token = self.cls_token.expand(batch_size, -1, -1)
    
        x = torch.cat((cls_token, x), dim=1)
    
        x = x + self.pos_embed
    
        return self.pos_drop(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class Mlp(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    
    def __init__(self, img_size=28, patch_size=7, in_chans=1, num_classes=10, embed_dim=128, depth=6, num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        self.pos_drop = ViTPositionalAndCls(self.patch_embed.n_patches, embed_dim, drop_rate)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x[:, 0]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class BasicVisionTransformer(nn.Module):
    
    def __init__(self, img_size=28, patch_size=7, in_chans=1, num_classes=10, 
                embed_dim=128, depth=6, num_heads=8, mlp_ratio=4., 
                qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def initialize_special_tokens(self):
        nn.init.trunc_normal_(self.pos_drop.pos_embed, std=.02)
        nn.init.trunc_normal_(self.pos_drop.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        outcome = self.head(cls_token_final)
        
        return outcome

if __name__ == "__main__":
    
    sample_img = torch.randn(1, 3, 224, 224)
    patch_emb = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
    output = patch_emb(sample_img)
    print(f"Input Shape: {sample_img.shape}")
    print(f"Output Shape: {output.shape}")
    
    
    dummy_patches = torch.randn(1, 196, 768)
    pos_layer = ViTPositionalAndCls()
    output = pos_layer(dummy_patches)
    print(f"Input Shape: {dummy_patches.shape}")
    print(f"Output Shape after CLS & POS: {output.shape}")
    
    
    dummy_input = torch.randn(1, 197, 768)
    mha = MultiHeadAttention(dim=768, num_heads=8)
    output = mha(dummy_input)
    print(f"MHA Output Shape: {output.shape}")
    
    model = BasicVisionTransformer(n_classes=10)
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_image)
    
    print(f"Final Model Structure: ViT-Base-Patch16")
    print(f"Input Shape: {dummy_image.shape}")
    print(f"Output Logits Shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")