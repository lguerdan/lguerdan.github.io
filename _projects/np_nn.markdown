---
layout: page
title: CNNs from scratch
description: A numpy CNN implementation for digit classification
img: /assets/img/np_cnn_preview.png
importance: 1
---

For a class project in my graduate level neural networks class in spring of 2019, I implemented a CNN for digit classification using numpy. I implemented an optimized [toeplitz matrix representation](https://en.wikipedia.org/wiki/Toeplitz_matrix) for convolutions and included this in a generalized CNN Python class that could be easily extended to different network structures.  

See the code for this implementation and several experiments [here](https://github.com/lguerdan/numpy-CNN), and my project report below. Note that the deconvolution demonstration at the end of the report has an issue that still needs to be fixed. 

<div>
  <iframe src="{{site.baseurl}}/assets/pdf/np-nn-2019-report.pdf" width="100%" height="800px"></iframe>
</div>
