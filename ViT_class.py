import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block, Attention



class ModifiedAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)  (batch_size, num_head, num_tokens + 1, emb_size/num_head)
        # print("k.transpose(-2, -1)", k.transpose(-2, -1).shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_head, num_tokens + 1, num_tokens + 1)
        # attn_r = attn[:, :, 0, :].mean(dim=1).squeeze()
        # attn_r = attn.mean(dim=1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        attn_r = attn.mean(dim=1)
        return x, attn_r


class ModifiedBlock(Block):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__(dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer)

        self.attn = ModifiedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x_attn, attn = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(x_attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, attn



class ModifieddVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(block_fn=ModifiedBlock, *args, **kwargs)


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with modifications
        B = x.shape[0]
        x = self.patch_embed(x)
        # print(x.shape)
        x = self._pos_embed(x)

        x = self.norm_pre(x)
        attns = []
        for i in range(len(self.blocks)):
            x, weights = self.blocks[i](x)
            attns.append(weights)

        x = self.norm(x)
        return x, attns

    def forward(self, x):

        x, attns = self.forward_features(x)

        x = self.forward_head(x)
        return x, attns

    # def check_func(self):
    #     for module in self.children():
    #         if hasattr(module, 'training'):
    #             print(f'{module.__class__.__name__}: training={module.training}')
    #     print(len(self.blocks))
        # pass
