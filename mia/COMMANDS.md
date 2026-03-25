# MIA Script Commands

以下命令均默认在项目根目录 `Collaborative-WSVAD` 下执行。

## 1. 重建成员与非成员清单

脚本: `mia/rebuild_ucf_event_membership.py`

只生成 `ucf-event`:

```powershell
python mia\rebuild_ucf_event_membership.py --dataset ucf
```

只生成 `xd-event`:

```powershell
python mia\rebuild_ucf_event_membership.py --dataset xd
```

同时生成两个数据集:

```powershell
python mia\rebuild_ucf_event_membership.py --dataset all
```

显式指定 checkpoint:

```powershell
python mia\rebuild_ucf_event_membership.py --dataset ucf --checkpoint PAMP-FedVAD/models/ucf-event_model_roc_0.8797.pth
```

```powershell
python mia\rebuild_ucf_event_membership.py --dataset xd --checkpoint PAMP-FedVAD/models/xd-event_model_final_ap_0.8243.pth
```

## 2. 提取攻击特征

脚本: `mia/extract_attack_features.py`

UCF 10-crop 特征提取:

```powershell
python mia\extract_attack_features.py --manifest mia\manifests\membership_manifest_ucf_event_10crop.csv --output mia\features\attack_features_ucf_event_10crop.csv
```

XD 10-crop 特征提取:

```powershell
python mia\extract_attack_features.py --manifest mia\manifests\membership_manifest_xd_event_10crop.csv --output mia\features\attack_features_xd_event_10crop.csv
```

UCF 小规模试跑:

```powershell
python mia\extract_attack_features.py --manifest mia\manifests\membership_manifest_ucf_event_10crop.csv --output mia\features\attack_features_ucf_event_10crop_smoke.csv --limit 10
```

XD 小规模试跑:

```powershell
python mia\extract_attack_features.py --manifest mia\manifests\membership_manifest_xd_event_10crop.csv --output mia\features\attack_features_xd_event_10crop_smoke.csv --limit 10
```

## 3. 训练 MIA 攻击模型

脚本: `mia/train_attack_model.py`

UCF:

```powershell
python mia\train_attack_model.py --features mia\features\attack_features_ucf_event_10crop.csv --name ucf_event_10crop
```

XD:

```powershell
python mia\train_attack_model.py --features mia\features\attack_features_xd_event_10crop.csv --name xd_event_10crop
```

## 4. 训练客户端归属推断模型

脚本: `mia/train_client_attribution_model.py`

UCF:

```powershell
python mia\train_client_attribution_model.py --features mia\features\attack_features_ucf_event_10crop.csv --name ucf_event_10crop
```

XD:

```powershell
python mia\train_client_attribution_model.py --features mia\features\attack_features_xd_event_10crop.csv --name xd_event_10crop
```

## 5. 绘制 MIA 混淆矩阵

脚本: `mia/plot_mia_confusion_matrices.py`

UCF 原始计数版:

```powershell
python mia\plot_mia_confusion_matrices.py --features mia\features\attack_features_ucf_event_10crop.csv --name ucf_event_10crop
```

XD 原始计数版:

```powershell
python mia\plot_mia_confusion_matrices.py --features mia\features\attack_features_xd_event_10crop.csv --name xd_event_10crop
```

UCF 归一化版:

```powershell
python mia\plot_mia_confusion_matrices.py --features mia\features\attack_features_ucf_event_10crop.csv --name ucf_event_10crop --normalize
```

XD 归一化版:

```powershell
python mia\plot_mia_confusion_matrices.py --features mia\features\attack_features_xd_event_10crop.csv --name xd_event_10crop --normalize
```

## 6. 绘制客户端归属推断混淆矩阵

脚本: `mia/plot_client_attribution_confusion_matrices.py`

UCF 原始计数版:

```powershell
python mia\plot_client_attribution_confusion_matrices.py --features mia\features\attack_features_ucf_event_10crop.csv --name ucf_event_10crop
```

XD 原始计数版:

```powershell
python mia\plot_client_attribution_confusion_matrices.py --features mia\features\attack_features_xd_event_10crop.csv --name xd_event_10crop
```

UCF 归一化版:

```powershell
python mia\plot_client_attribution_confusion_matrices.py --features mia\features\attack_features_ucf_event_10crop.csv --name ucf_event_10crop --normalize
```

XD 归一化版:

```powershell
python mia\plot_client_attribution_confusion_matrices.py --features mia\features\attack_features_xd_event_10crop.csv --name xd_event_10crop --normalize
```
