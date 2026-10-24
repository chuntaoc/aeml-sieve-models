"""Export SAM 2 / SAM 2.1 to ONNX with the AEML, NTUST SAM contract, split into parts for GitHub + jsDelivr.

Usage (on a computer with internet access):
    pip install torch onnx onnxruntime onnxsim hydra-core iopath
    git clone https://github.com/facebookresearch/sam2.git && cd sam2 && SAM2_BUILD_CUDA=0 pip install -e .
    # download a checkpoint, e.g. sam2.1_hiera_tiny.pt (see the sam2 README)
    python export_sam2.py --cfg configs/sam2.1/sam2.1_hiera_t.yaml --ckpt sam2.1_hiera_tiny.pt --out sam2.1_tiny

Contract (same as the MobileSAM pack, so the browser code is shared):
    encoder: image [1,3,1024,1024] (normalised with ImageNet mean/std x 255)
             -> image_embed [1,256,64,64], high_res_feats_0 [1,32,256,256], high_res_feats_1 [1,64,128,128]
    decoder: image_embed, high_res_feats_0, high_res_feats_1, point_coords [B,1,2] (px in the 1024 input),
             point_labels [B,1] -> low_res_masks [B,3,256,256] (logits), iou [B,3]
(c) 2026 AEML, NTUST. SAM 2 itself is (c) Meta Platforms, Inc., Apache-2.0.
"""
import argparse, os, json, hashlib
import numpy as np
import torch
from sam2.build_sam import build_sam2

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', default='configs/sam2.1/sam2.1_hiera_t.yaml')
ap.add_argument('--ckpt', default=None, help='checkpoint (.pt); omit only for a structural test with random weights')
ap.add_argument('--out', default='sam2.1_tiny')
ap.add_argument('--chunk_mb', type=float, default=15)
a = ap.parse_args()

model = build_sam2(a.cfg, a.ckpt, device='cpu', mode='eval')
model.eval()
FEAT = [(256, 256), (128, 128), (64, 64)]


class Enc(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m

    def forward(s, image):
        bo = s.m.forward_image(image)
        _, vf, _, _ = s.m._prepare_backbone_features(bo)
        if s.m.directly_add_no_mem_embed:
            vf[-1] = vf[-1] + s.m.no_mem_embed
        feats = [f.permute(1, 2, 0).reshape(1, -1, *sz) for f, sz in zip(vf[::-1], FEAT[::-1])][::-1]
        return feats[-1], feats[0], feats[1]


class Dec(torch.nn.Module):
    """Prompt encoding written out explicitly (a padding point with label -1 is appended, as SAM 2 does)."""
    def __init__(s, m): super().__init__(); s.m = m

    def forward(s, image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels):
        pe = s.m.sam_prompt_encoder
        B = point_coords.shape[0]
        coords = torch.cat([point_coords, torch.zeros_like(point_coords[:, :1, :])], dim=1) + 0.5
        labels = torch.cat([point_labels, -torch.ones_like(point_labels[:, :1])], dim=1).unsqueeze(-1)
        emb = pe.pe_layer._pe_encoding(coords / 1024.0)
        emb = emb * (labels != -1).float() + pe.not_a_point_embed.weight * (labels == -1).float()
        emb = emb + pe.point_embeddings[0].weight * (labels == 0).float() + pe.point_embeddings[1].weight * (labels == 1).float()
        dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(B, -1, 64, 64)
        out = s.m.sam_mask_decoder(image_embeddings=image_embed, image_pe=pe.get_dense_pe(), sparse_prompt_embeddings=emb,
                                   dense_prompt_embeddings=dense, multimask_output=True, repeat_image=True,
                                   high_res_features=[high_res_feats_0, high_res_feats_1])
        return out[0], out[1]


class DecRef(torch.nn.Module):
    """Reference path through the official prompt encoder (for the parity check)."""
    def __init__(s, m): super().__init__(); s.m = m

    def forward(s, e, h0, h1, pc, pl):
        sp, de = s.m.sam_prompt_encoder(points=(pc, pl), boxes=None, masks=None)
        out = s.m.sam_mask_decoder(image_embeddings=e, image_pe=s.m.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sp,
                                   dense_prompt_embeddings=de, multimask_output=True, repeat_image=True, high_res_features=[h0, h1])
        return out[0], out[1]


os.makedirs(a.out, exist_ok=True)
enc_f, dec_f = os.path.join(a.out, 'sam2_encoder.onnx'), os.path.join(a.out, 'sam2_decoder.onnx')
x = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    e, h0, h1 = Enc(model)(x)
    torch.onnx.export(Enc(model), (x,), enc_f, input_names=['image'], output_names=['image_embed', 'high_res_feats_0', 'high_res_feats_1'],
                      opset_version=17, dynamo=False)
    pc = torch.rand(5, 1, 2) * 1024
    pl = torch.ones(5, 1)
    torch.onnx.export(Dec(model), (e, h0, h1, pc, pl), dec_f,
                      input_names=['image_embed', 'high_res_feats_0', 'high_res_feats_1', 'point_coords', 'point_labels'],
                      output_names=['low_res_masks', 'iou'],
                      dynamic_axes={'point_coords': {0: 'B'}, 'point_labels': {0: 'B'}, 'low_res_masks': {0: 'B'}, 'iou': {0: 'B'}},
                      opset_version=17, dynamo=False)

# drop intermediate shape annotations: tracing leaves a stale declared shape in the Hiera trunk
# that onnxruntime-web rejects ("Can't merge shape info"); shapes are re-inferred at load time
# and fold the dynamic shape arithmetic with onnx-simplifier (pip install onnxsim); without this,
# onnxruntime-web refuses the SAM 2 encoder ("Can't merge shape info ... Inferred=4 Declared=5").
import onnx
try:
    import onnxsim
except ImportError:
    onnxsim = None
    print('WARNING: pip install onnxsim  (needed for onnxruntime-web)')
for fn in (enc_f, dec_f):
    mo = onnx.load(fn)
    del mo.graph.value_info[:]
    if onnxsim is not None and fn == enc_f:
        mo, ok = onnxsim.simplify(mo)
        print('onnxsim', os.path.basename(fn), 'ok' if ok else 'check failed')
    onnx.save(mo, fn)

import onnxruntime as ort
se, sd = ort.InferenceSession(enc_f), ort.InferenceSession(dec_f)
eo = se.run(None, {'image': x.numpy()})
print('encoder max |diff|:', max(float(np.abs(eo[i] - t.numpy()).max()) for i, t in enumerate([e, h0, h1])))
with torch.no_grad():
    lr, ir = DecRef(model)(e, h0, h1, pc, pl.long())
lo, io = sd.run(None, {'image_embed': e.numpy(), 'high_res_feats_0': h0.numpy(), 'high_res_feats_1': h1.numpy(),
                       'point_coords': pc.numpy(), 'point_labels': pl.numpy()})
print('decoder max |diff| masks:', float(np.abs(lo - lr.numpy()).max()), ' iou:', float(np.abs(io - ir.numpy()).max()), ' shapes', lo.shape, io.shape)

# split into parts + manifest
CH = int(a.chunk_mb * 1024 * 1024)
man = {'name': os.path.basename(a.out), 'family': 'sam2', 'input': 1024, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375],
       'contract': 'see export_sam2.py', 'license': 'Apache-2.0 (SAM 2, (c) Meta Platforms, Inc.)', 'checkpoint': os.path.basename(a.ckpt or 'random-init')}
for part, fn in [('encoder', enc_f), ('decoder', dec_f)]:
    b = open(fn, 'rb').read()
    parts = []
    for i in range(0, len(b), CH):
        pn = os.path.basename(fn) + '.part%d' % (i // CH)
        open(os.path.join(a.out, pn), 'wb').write(b[i:i + CH])
        parts.append(pn)
    man[part] = {'parts': parts, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
    os.remove(fn)
json.dump(man, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=2)
print('pack written to', a.out, {k: man[k]['parts'] for k in ('encoder', 'decoder')})
