---
layout: distill
title: "TurboQuant 1 - 初识"
date: "2026-04-08 21:15:00"
description: 推理笔记
featured: true

authors:
  - name: Shuliang Li
    url: "https://l1351868270.github.io/"
    affiliations:
      name: MIT CSAIL

bibliography: 2026-03-05-po.bib

toc:
  - name: TurboQuant
    subsections:
        - name: 极坐标
---

# 极坐标
## 直角坐标 -> 极坐标

$$r = \sqrt{x^{2} + y^{2}}$$

$$\text{tan}(\theta) = \frac{y}{x}$$

## 极坐标 -> 直角坐标

$$x = r * \text{cos}(\theta)$$

$$y = r * \text{sin}(\theta)$$

# fact1
For any positive integer $d$, if $x \in R^{d}$ is a zero mean unit variance isotropic Gaussian
random variable in dimension $d$, i.e., $x ∼ N (0, I_d)$, then its 2-norm, denoted by $r := \lVert x_2 \rVert$, follows
a generalized gamma distribution with the following probability density for any $r ≥ 0$:

$$f_{R}\left( r \right) = \frac{2}{2^{d/2} \cdot \Gamma \left( d / 2 \right)} r^{d-1} exp \left(-r^{2}/2 \right)$$


## 验证
{% include figure.liquid 
   path="assets/img/2026-04-09-turboquant-3/1.png" 
   class="img-fluid rounded z-depth-1" 
   title="图片标题" 
   caption="" 
%}
```
import torch
from torch.distributions import Gamma, TransformedDistribution
from torch.distributions.transforms import PowerTransform
import matplotlib.pyplot as plt

dims = [16, 64, 256, 1024]
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
plt.figure(figsize=(12, 7))

for i, dim in enumerate(dims):
    # 1. 定义 Chi 分布
    # Chi2(df) 等价于 Gamma(df/2, 0.5)
    base_dist = Gamma(torch.tensor([dim / 2.0]), torch.tensor([0.5]))
    chi_dist = TransformedDistribution(base_dist, PowerTransform(0.5))
    
    # 2. 采样并计算平方模长 ||x||^2
    x_samples = torch.randn(n_samples, dim)
    # 计算平方和：这才是严格符合 Gamma(d/2, 0.5) 的变量
    samples = torch.sum(x_samples**2, dim=1).sqrt()

    
    # 3. 绘制直方图 (使用 density=True 确保 y 轴为频率密度)
    plt.hist(samples.numpy(), bins=40, alpha=0.5, density=True, 
             label=f'df={dim}', color=colors[i], edgecolor='white')
    
    # 4. 绘制拟合的概率密度曲线 (PDF)
    # 根据采样范围动态生成 x 轴
    x_min, x_max = samples.min() * 0.8, samples.max() * 1.2
    x = torch.linspace(x_min, x_max, 500)
    pdf = chi_dist.log_prob(x).exp()
    plt.plot(x.numpy(), pdf.numpy(), color=colors[i], lw=2)

plt.title('Chi Distribution Sampling (1000 samples per df)', fontsize=15)
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()
```