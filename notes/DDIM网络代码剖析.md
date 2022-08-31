# 1. models

## diffusion.py

​	

```python
torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
```

分组做Normalization，计算公式如下：

![img](https://img-blog.csdnimg.cn/0b66484c529e48ac9a78549db98ad3da.png)

E[x]是x的均值；
Var[x]是标准差；
$\gamma$和$\beta$是训练参数，如果不想使用，可以通过参数affine=False设置。默认为True；
$\epsilon$是输入参数，防止Var为0，默认值为*1e-05*,可以通过参数eps修改。

```
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
```

根据给定的size或scale_factor参数来对输入进行下/上采样，使用的插值算法取决于参数mode的设置。

- input (Tensor) – 输入张量

- size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) – 输出大小.
- scale_factor (float or Tuple[float]) – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型
- mode (str) – 可使用的上采样算法，有’nearest’, ‘linear’, ‘bilinear’, ‘bicubic’ , ‘trilinear’和’area’. 默认使用’nearest’
- align_corners (bool, optional) – 几何上，我们认为输入和输出的像素是正方形，而不是点。如果设置为True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。如果设置为False，则输入和输出张量由它们的角像素的角点对齐，插值使用边界外值的边值填充;当scale_factor保持不变时，使该操作独立于输入大小。仅当使用的算法为’linear’, ‘bilinear’, 'bilinear’or 'trilinear’时可以使用。默认设置为False
    
