"""Export MobileSAM to ONNX with the AEML, NTUST SAM contract.
encoder: image [1,3,1024,1024] (normalised) -> image_embed [1,256,64,64]
decoder: image_embed, point_coords [B,1,2] (1024-input px), point_labels [B,1]
         -> low_res_masks [B,3,256,256] (logits), iou [B,3]
"""
import torch, numpy as np, sys
from mobile_sam import sam_model_registry
ck = sys.argv[1] if len(sys.argv) > 1 else 'mobile_sam.pt'   # from https://github.com/ChaoningZhang/MobileSAM (weights/)
sam = sam_model_registry['vit_t'](checkpoint=ck).eval()

class Enc(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, image): return s.m.image_encoder(image)

class Dec(torch.nn.Module):
    """Prompt encoding written out explicitly (as in SAM's ONNX export) so that a batch of
    single foreground points traces cleanly; a padding point (label -1) is appended as SAM does."""
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, image_embed, point_coords, point_labels):
        pe = s.m.prompt_encoder
        B = point_coords.shape[0]
        pad = torch.zeros_like(point_coords[:, :1, :])
        coords = torch.cat([point_coords, pad], dim=1) + 0.5
        labels = torch.cat([point_labels, -torch.ones_like(point_labels[:, :1])], dim=1)
        coords = coords / 1024.0
        emb = pe.pe_layer._pe_encoding(coords)
        lab = labels.unsqueeze(-1)
        emb = emb * (lab != -1).float() + pe.not_a_point_embed.weight * (lab == -1).float()
        emb = emb + pe.point_embeddings[0].weight * (lab == 0).float() + pe.point_embeddings[1].weight * (lab == 1).float()
        dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(B, -1, 64, 64)
        low, iou = s.m.mask_decoder(image_embeddings=image_embed, image_pe=pe.get_dense_pe(),
                                    sparse_prompt_embeddings=emb, dense_prompt_embeddings=dense, multimask_output=True)
        return low, iou

class DecRef(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, image_embed, point_coords, point_labels):
        sp, de = s.m.prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)
        return s.m.mask_decoder(image_embeddings=image_embed, image_pe=s.m.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=True)

x = torch.randn(1, 3, 1024, 1024)
torch.onnx.export(Enc(sam), (x,), 'mobilesam_encoder.onnx', input_names=['image'], output_names=['image_embed'], opset_version=17, dynamo=False)
emb = torch.randn(1, 256, 64, 64); pc = torch.rand(4, 1, 2) * 1024; pl = torch.ones(4, 1, dtype=torch.int64)
torch.onnx.export(Dec(sam), (emb, pc, pl.float()), 'mobilesam_decoder.onnx', input_names=['image_embed', 'point_coords', 'point_labels'],
                  output_names=['low_res_masks', 'iou'], dynamic_axes={'point_coords': {0: 'B'}, 'point_labels': {0: 'B'}, 'low_res_masks': {0: 'B'}, 'iou': {0: 'B'}},
                  opset_version=17, dynamo=False)
import onnxruntime as ort
with torch.no_grad():
    e_t = sam.image_encoder(x); l_t, i_t = DecRef(sam)(e_t, pc, pl)
se = ort.InferenceSession('mobilesam_encoder.onnx'); sd = ort.InferenceSession('mobilesam_decoder.onnx')
e_o = se.run(None, {'image': x.numpy()})[0]
l_o, i_o = sd.run(None, {'image_embed': e_o, 'point_coords': pc.numpy(), 'point_labels': pl.float().numpy()})
print('enc diff', np.abs(e_o - e_t.numpy()).max(), 'dec mask diff', np.abs(l_o - l_t.numpy()).max(), 'iou diff', np.abs(i_o - i_t.numpy()).max())
print('shapes', e_o.shape, l_o.shape, i_o.shape)
