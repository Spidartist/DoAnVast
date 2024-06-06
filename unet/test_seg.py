from model.vit import vit_base
from model.segmenter_decoder import MaskTransformer, Segmenter


n_cls = 2
encoder = vit_base(img_size=[256])
decoder = MaskTransformer(n_cls=n_cls, patch_size=encoder.patch_embed.patch_size, 
                          d_encoder=encoder.embed_dim,
                          n_layers=2, n_heads=encoder.embed_dim//64, mlp_ratio=4.,
                          d_model=encoder.embed_dim,
                          drop_path_rate=0.0,
                          dropout=0.1
                          )

model = Segmenter(encoder=encoder, decoder=decoder, n_cls=n_cls)

