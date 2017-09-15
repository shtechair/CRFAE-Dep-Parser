# CRF autoencoder for unsupervised dependency parsing

This repository contains the code to reproduce the experiment result of the paper
[CRF autoencoder for unsupervised dependency parsing](http://sist.shanghaitech.edu.cn/faculty/tukw/emnlp17CJT.pdf) on WSJ data set and PASCAL dataset.



## Prerequisites
1. Java 8
2. sbt 0.13.13

## Compile
```
sbt assembly
```

## Example Usage
1. Experiments on WSJ dataset.
```
java -cp target/scala-2.12/CRFAE-Dep-Parser-assembly-1.0.jar edu.shanghaitech.nlp.crfae.parser.DepParserLauncher --train-file wsj10-train.txt --test-file wsj-test.txt --model-type projective --training-type hard --reg-type L1 --rules-type wsj --km-type decoder --gd-num-passes 10 --em-num-passes 10 --batch-size 200 --init-rate 0.1 --lambda 2.5 --prior-weight 9.0
```

2. Experiments on Pascal dataset.

```
java -cp target/scala-2.12/CRFAE-Dep-Parser-assembly-1.0.jar edu.shanghaitech.nlp.crfae.parser.DepParserLauncher --train-file pascal-{}-train.txt --test-file pascal-{}-test.txt --model-type projective --training-type hard --reg-type L1 --rules-type ud --km-type joint --gd-num-passes 2 --em-num-passes 2 --batch-size 200 --init-rate {} --lambda {} --prior-weight {}
```


| language | learning rate (init-rate) | lambda | prior weight|
| ------------- | ------------- | ---------- |---------- | 
| basque | 0.01 | 2.5  | 0.1 |
|  czech |  0.01 | 1.0 |  1.0 |
| danish | 0.05 | 2.5 | 25.0| 
|  dutch | 0.1 | 2.5 | 9.0 |
| protuguese | 0.05 | 2.5 | 5.0|
|  slovene | 0.01 | 0.5 | 1.0 |
|  swedish | 0.05 | 0.5 | 25.0 |

## Input format

```
No	wonder	
DT	NN	
DT	NN	
2	0	

So	he	adjusts	
RB	PRP	VBZ	
RB	PRP	VBZ	
3	3	0	
```

1. Sequence of words.
2. Sequence of POS.
3. Sequence of Universal POS. (If it is not available, duplicate the sequence of POS.)
4. Sequence of Head index.



## Citation
If you found this repository helpful, you could cite
```
@inproceedings{cai2017crf,
  title={CRF Autoencoder for Unsupervised Dependency Parsing},
  author={Cai, Jiong and Jiang, Yong and Tu, Kewei},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={1639--1644},
  year={2017}
}
```

## Licence
This code is distributed under GPLv3.0 LICENSE.
