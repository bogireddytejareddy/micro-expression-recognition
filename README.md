# Spontaneous Facial Micro Expression Recognition using 3D Spatio-Temporal Convolutional Neural Networks

## Abstract

Facial expression recognition in videos is an active area of research in computer vision. However, fake facial expressions are difficult to be recognized even by humans. On the other hand, facial micro-expressions generally represent the actual emotion of a person, as it is a spontaneous reaction expressed through human face. Despite of a few attempts made for recognizing micro-expressions, still the problem is far from being a solved problem, which is depicted by the poor rate of accuracy shown by the state-of-the-art methods. A few CNN based approaches are found in the literature to recognize micro-facial expressions from still images. Whereas, a spontaneous micro-expression video contains multiple frames that have to be processed together to encode both spatial and temporal information. This paper proposes two 3D-CNN methods: MicroExpSTCNN and MicroExpFuseNet, for spontaneous facial micro-expression recognition by exploiting the spatiotemporal information in CNN framework. The MicroExpSTCNN considers the full spatial information, whereas the MicroExpFuseNet is based on the 3D-CNN feature fusion of the eyes and mouth regions. The experiments are performed over CAS(ME)^2 and SMIC micro-expression databases. The proposed MicroExpSTCNN model outperforms the state-of-the-art methods.

## Prerequisites
- [Keras 2.0.0](https://github.com/fchollet/keras) Strictly

## Results
| Method | Proposed Year | Method Type | CAS(ME)^2 | SMIC |
| ------ | ------------- | ----------- | --------- | ---- |
| LBP-TOP |     2013          |     HCM        |     -      |   42.72%   |
|  STCLQP  |        2016       |    HCM         |     -      |  64.02%    |
| CNN with Augumentation       |     2017          |   DLM          |    78.02%       |  -    |
|  3D-FCNN  |   2018            |    DLM         |    -       |   55.49%   |
|   MicroExpSTCNN     |   Proposed            |   DLM          |      87.80%     |   68.75%   |
|  Intermediate MicroExpFuseNet     |  Proposed             |   DLM          |    83.25%       |   54.77%   |
|  Late MicroExpFuseNet   |    Proposed           |    DLM         |    79.31%       |  64.82%    |

## Validation Data and Weights
 ### [CASME-SQUARE](https://drive.google.com/file/d/1v5v_W-N-CslBgiwNdww8_QF5TAWgjE_c/view?usp=sharing)
 ### [SMIC](https://drive.google.com/file/d/1hotsk5TSnSxuLHqHC990wMYox1L1Vj6Q/view?usp=sharing)

## Citation

If you use this code in your research, we would appreciate a citation:

	@article{reddy2019microexpression,
            title={Spontaneous Facial Micro-Expression Recognition using 3D Spatiotemporal Convolutional Neural Networks},
            author={Sai Prasanna Teja, Reddy and Surya Teja, Karri and Dubey, Shiv Ram and Mukherjee, Snehasis},
            journal={International Joint Conference on Neural Networks},
            year={2019}
            }
## License

Copyright (c) 2019 Bogireddy Teja Reddy. Released under the MIT License. See [LICENSE](LICENSE) for details.
