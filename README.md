# ScissionTL: A Benchmarking Tool for Accelerating Distributed Deep Neural Networks Using Transfer Layer for Edge Computing

## About the research

<img src="images/overall_design.png" alt="overall design" width="500"/>

ScissionTL is a benchmarking tool for distributing deep neural networks (DNNs) across multiple resource tiers, such as the device, edge and cloud. ScissionTL is based on Scission that automatically benchmarks DNN models on a target set of the device, edge, and cloud resources for deciding the optimal slicing point for maximizing inference performance. 

The original Scission source code is available "[here](https://github.com/qub-blesson/scission#scission-a-tool-for-maximising-performance-of-deep-neural-networks-in-edge-computing)". The research article describing Scission can be cited as follows: 

Luke Lockhart, Paul Harvey, Pierre Imai, and Blesson Varghese, "[Scission: Performance-driven and Context-aware Cloud-Edge Distribution of Deep Neural Networks](https://arxiv.org/pdf/2008.03523.pdf)", IEEE/ACM International Conference on Utility and Cloud Computing, 2020. 

Based on Scission, ScissionTL inserts an additional traffic-aware layer called the Transfer Layer (TL) between the split point of the DNN. The TL is a small neural network layer that can reduce the amount of data exchanged during communication. The TL is composed of the device transfer layer (DeviceTL) and edge transfer layer (EdgeTL) layers respectively for the device and edge resources. The DeviceTL layer compresses the feature maps of the end layer of the sliced DNN on the mobile device. The compressed data is then transferred through the network connection. The EdgeTL layer expands the received data and passes the data to the starting layer of the remaining DNN on the edge. ScissionTL decides the model’s optimal slicing point when the TL is applied. We developed two versions of ScissionTL for the edge by using TensorFlow and NVIDIA Triton.

<img src="images/densenet.png" alt="densenet" width="750"/>


### Citing the research on ScissionTL

Hyunho Ahn, Munkyu Lee, Cheol-Ho Hong, and Blesson Varghese, "[ScissionLite: Accelerating Distributed Deep Neural Networks Using Transfer Layer](https://arxiv.org/pdf/2105.02019.pdf)", arXiv e-prints (2021): arXiv-2105.


