# Dataset class for TensorFlow

## 必要なライブラリ

* os
* numpy
* skimage

## ディレクトリ構造

./dataset
├── label_a
├── label_b
└── label_c

## 使い方

```
md = MakeDataset(num_train=500)
train_img, train_label, valid_img, valid_label = md.LoadingData()
print(md())

Result:
# ==========================================
# debug :
#    train :
#        images.shape : (500, 75, 75, 3)
#        labels.shape : (500, 3)
#    valid :
#        images.shape : (100, 75, 75, 3)
#        labels.shape : (100, 3)
# Num all data   : 600
# Num train data : 500
# Num valid data : 100
# ==========================================

['label_a', 'label_b', 'label_c']
```