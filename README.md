# ATL-Net
Learning Task-aware Local Representations for Few-shot Learning, IJCAI 2020

## Prerequisites
- Python 3
- PyTorch 1.4.0


## DataSets
Please refer to [DN4](https://github.com/WenbinLee/DN4).

##  Train & Test
DataSet is miniImagenet, CUB, StanfordCar or StanfordDog.

- Train:
```bash
python -u trainer.py -c ./config/${DataSet}_Conv64F_5way_1shot.json -d 0
python -u trainer.py -c ./config/${DataSet}_Conv64F_5way_5shot.json -d 0
```
- Test:
```bash
python -u test.py -r ./results/${DataSet}_Conv64F_5way_1shot -d 0
python -u test.py -r ./results/${DataSet}_Conv64F_5way_5shot -d 0
```


## Note
Sorry about the mistakes in the Eq.(4) and the Eq.(7), 
the Eq.(4) is a step function, and Eq.(7) is the approximation of the Eq.(4) with the adaptive threshold,
both of them repeatedly introduce the process of the Eq.(5),
the paper after correction is [here](https://github.com/LegenDong/ATL-Net/blob/master/pdf/ATL-Net_Update.pdf).
![avatar](./images/correction.jpg)


## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{ijcai2020-100,
  title     = {Learning Task-aware Local Representations for Few-shot Learning},
  author    = {Dong, Chuanqi and Li, Wenbin and Huo, Jing and Gu, Zheng and Gao, Yang},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {716--722},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/100},
  url       = {https://doi.org/10.24963/ijcai.2020/100},
}
```

## Reference
Our code is based on [DN4](https://github.com/WenbinLee/DN4).

