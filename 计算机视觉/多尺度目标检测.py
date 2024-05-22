import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

img = d2l.plt.imread('./catdog.jpg')
h, w = img.shape[:2]


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    # 这个函数实际上时做了一个小trick，这里传入的fmap则是将原始图片映射后的大小，在映射后的图片上生成锚框后，因为返回的是一个比例，乘原始图片的高与宽之后，则变化为对应的锚框
    # 这里传入的h，w即为每行每列生成的锚框的个数
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

display_anchors(fmap_w=5, fmap_h=4, s=[0.75, 0.5, 0.25])
plt.show()
