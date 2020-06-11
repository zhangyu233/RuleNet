# Jointly Differentiable Learning of Embeddings and Logic Rules for Knowledge Graph Reasoning

**Introduction**

This this the Pytorch implementation of the [RuleNet]. 

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

 **Train**

 For example, this command trains a RuleNet model on FB15k dataset with GPU 0.
 ~~~
 python main.py --do_train --do_valid --do_test  --device cuda:0 --data_path ./data/FB15k --max_steps 100000 --d 300 --b 300 -lr 0.0001 --save ./save/FB15k_model
 ~~~

 **Test**

 The pretrained models can be downloaded at:
~~~
https://drive.google.com/drive/folders/15cRtMk7URLCqmN5_cJoF3mrMXc_mDuRb?usp=sharing
~~~

The model can be evaluated using the following metrics MRR, MR, HITS@1, HITS@3, HITS@10. This command test a pretrained model on FB15k dataset with GPU 0:

~~~
 python main.py  --do_test --device cuda:0 --data_path ./data/FB15k --save_path ./save/FB15k_model --init_checkpoint ./save/FB15knew
~~~

Rule extraction

The logic rules can be recovered using the following command:

~~~
python main.py  --device cuda:0  --recover_rule  --data_path ./data/FB15k --save_path ./save/FB15knew --init_checkpoint ./save/FB15knew
~~~