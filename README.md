# aeml-sieve-models

AI models for the **AEML, NTUST Fine Aggregate Sieve Analysis Tool** (https://www.concrete.tw).
先進工程材料研究室（AEML），國立台灣科技大學營建工程系 — 混凝土細粒料篩分析工具使用的 AI 模型。
所有模型都在使用者的瀏覽器內執行（onnxruntime-web），照片不會上傳。

## 內容 / Contents

| 路徑 | 說明 | 大小 |
|---|---|---|
| `stardist_2d_versatile_fluo.onnx` | StarDist 2D_versatile_fluo（ONNX，opset 17）。輸入 `input` [1,H,W,1]；輸出 `prob` [1,H/2,W/2,1]、`dist` [1,H/2,W/2,32] | 5.7 MB |
| `mobilesam/` | MobileSAM（vit_t）模型包：`manifest.json` + encoder／decoder 分割檔（每檔 < 20 MB，供 jsDelivr 提供） | 約 45 MB |
| `sam2.1_tiny/` | SAM 2.1 tiny 模型包（依該資料夾 README 自行產生後上傳） | 約 126 MB |
| `tools/` | 轉換程式：`convert_stardist_to_onnx.py`、`export_mobilesam.py`、`export_sam2.py`（含數值一致性檢查） | |

SHA-256 (stardist_2d_versatile_fluo.onnx): `77998841d88a50bebf80f4b5401fc23525c173f5463989df5fa187d334d4d9b1`

### SAM 模型包介面（MobileSAM 與 SAM 2 共用）
- encoder：`image` [1,3,1024,1024]（RGB，減平均除標準差）→ `image_embed` [1,256,64,64]（SAM 2 另有 `high_res_feats_0`、`high_res_feats_1`）
- decoder：encoder 全部輸出 + `point_coords` [B,1,2]（1024 輸入像素座標）+ `point_labels` [B,1] → `low_res_masks` [B,3,256,256]（logits）、`iou` [B,3]

## 部署 / Deployment
1. 本 repo 設為 **Public**，上傳全部檔案後建立 Release，tag 為 `v1`。
2. 工具預設網址（已內建）：
   - StarDist：`https://cdn.jsdelivr.net/gh/chuntaoc/aeml-sieve-models@v1/stardist_2d_versatile_fluo.onnx`
   - MobileSAM：`https://cdn.jsdelivr.net/gh/chuntaoc/aeml-sieve-models@v1/mobilesam/manifest.json`
3. 之後更新模型請用新 tag（`v2`…），jsDelivr 會快取同一 tag 的內容。

## 授權 / Licences
- StarDist — BSD 3-Clause, Copyright (c) 2018-2025 Uwe Schmidt, Martin Weigert (`LICENSE-StarDist.txt`). Schmidt et al., MICCAI 2018; Weigert et al., WACV 2020.
- MobileSAM — Apache-2.0, (c) Chaoning Zhang et al. (`mobilesam/LICENSE-MobileSAM.txt`). Zhang et al., *Faster Segment Anything*, 2023.
- SAM 2 — Apache-2.0, (c) Meta Platforms, Inc. (`sam2.1_tiny/LICENSE-SAM2.txt`). Ravi et al., *SAM 2*, 2024.
- 轉換程式 (tools/) — (c) 2026 AEML, NTUST.
- 篩分析工具本身不在此 repo：(c) 2026 AEML, NTUST. All rights reserved.

這些模型都未針對細粒料訓練，結果僅供研究與教學參考。
