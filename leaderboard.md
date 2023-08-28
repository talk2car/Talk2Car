# Talk2Car Leaderboard

One can find the current Talk2Car leaderboard here. The models on Talk2Car are evaluated by checking if the Intersection over Union of the predicted object bounding box and the ground truth bounding box is above 0.5.
This metric can be referred to by many ways i.e. IoU<sub>0.5</sub>, AP50, ...
Pull requests with new results and models are always welcome!

<div align="center">

| Model  | AP50 / IoU<sub>0.5</sub> | Code |
|:---:|:---:|:---:|
| [STACK-NMN](https://arxiv.org/pdf/1807.08556.pdf)  | 33.71  | |
| [SCRC](https://arxiv.org/abs/1511.04164)  | 38.7  | |
| [OSM](https://arxiv.org/pdf/1406.5679.pdf)  | 35.31  | | 
| [Bi-Directional retr.](https://arxiv.org/abs/2004.13822)  | 44.1  | | 
| [MAC](https://arxiv.org/abs/1803.03067)  |  50.51 | | 
| [MSRR](https://arxiv.org/abs/2003.08717) | 60.04 | |
| [VL-Bert (Base)](https://arxiv.org/abs/1908.08530)| 63.1 | [Code](https://github.com/ThierryDeruyttere/VL-BERT-Talk2Car) |
| [AttnGrounder](https://arxiv.org/abs/2009.05684) | 63.3 |[Code](https://github.com/i-m-vivek/AttnGrounder) |
| [ASSMR](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_5) | 66.0 | |
| [CMSVG](https://arxiv.org/abs/2009.06066) | 68.6 | [Code](https://github.com/niveditarufus/CMSVG) |
| [Vilbert (Base)](https://arxiv.org/abs/1908.02265) | 68.9| [Code](https://github.com/ThierryDeruyttere/vilbert-Talk2car) |
| [CMRT](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_3) | 69.1 | |
| [Sentence-BERT+FCOS3D](https://www.aaai.org/AAAI22Papers/AAAI-8858.GrujicicD.pdf) | 70.1 | |
| [Stacked VLBert](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_2) | 71.0 | |
| [FA](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9961196) | 73.51 |
Additional details about some of the baselines and state-of-the-art models mentioned in the leaderboard are also analysed in the C4AV challenge summary paper found [here](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_1).

</div>
