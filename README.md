This repository is the Python implementation for the following paper:
> Guosheng Kang, Linghang Ding, Jianxun Liu, Buqing Cao, and Yu Xu. Web API Recommendation based on Self-Attentional Neural Factorization Machines with Domain Interactions. IEEE Transactions on Network Science and Engineering, 2023.

## Environment Settings

* Python == 3.9.13
* Pandas == 1.4.4
* Numpy == 1.21.5
* Pytorch == 1.13.0
* torchvision == 0.14.0
* scikit-learn == 1.0.2


## Parameter Settings

- epoch: the number of epochs to train
- lr: learning rate
- embed_dim: embedding dimension
- droprate: dropout rate
- batch_size: batch size for training
- test_size: the percentage of test dataset
- endure_count: the threshold for ending training

## Files
- [SANFM_DI.py](SANFM_DI.py): training the SANFM_DI model and obtaining the test results
- [input_bert_data.csv](https://pan.baidu.com/s/1WC11FVhe6sxQYzihOiykYA?pwd=kang): the input dataset which consists of four domains, including the Mashup description document processed by the BERT model, Web API description document processed by the BERT model, Mashup category, and Web API category.


## Command to Run

~~~
python SANFM_DI.py 
~~~

**Note**: The dataset used in this code currently only divides into four domains. The sourse code needs to be modified as appropriate when use new dataset.  
**Contact**: linghangding@gmail.com (Linghang Ding)
