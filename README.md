# Contrastive Lifelong Learning for Anomaly Detection of Breast Cancer
====
source code of Contrastive Lifelong Learning for Anomaly Detection of Breast Cancer
----
Requirements
----
```python
pyhton >= 3.8.0
torch >= 1.9.0
torchvision >= 0.10.0
```

Train on Custom Dataset
----
To train the model on a custom dataset,
```python
Custom Dataset
├── test
│   ├── 0
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```
