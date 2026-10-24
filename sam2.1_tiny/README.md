# SAM 2.1 tiny pack (to be generated)

此資料夾預留給 SAM 2.1 hiera-tiny 模型包。產生方式（需可連網的電腦）：

```bash
pip install torch onnx onnxruntime onnxsim hydra-core iopath
git clone https://github.com/facebookresearch/sam2.git && cd sam2 && SAM2_BUILD_CUDA=0 pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ..        # 或依 sam2 README 下載 sam2.1_hiera_tiny.pt
python ../aeml-sieve-models/tools/export_sam2.py --cfg configs/sam2.1/sam2.1_hiera_t.yaml \
       --ckpt checkpoints/sam2.1_hiera_tiny.pt --out ../aeml-sieve-models/sam2.1_tiny
```

會產生 `manifest.json`、`sam2_encoder.onnx.part0…6`（約 109 MB）、`sam2_decoder.onnx.part0…1`（約 16.5 MB）。
上傳後在工具「SAM 模型包」貼上：
`https://cdn.jsdelivr.net/gh/chuntaoc/aeml-sieve-models@v2/sam2.1_tiny/manifest.json`（以實際 tag 為準）。

SAM 2: Apache-2.0, (c) Meta Platforms, Inc. — see LICENSE-SAM2.txt.
