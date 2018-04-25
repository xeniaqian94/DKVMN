# DKVMN

Dynamic Key-Value Memory Networks for Knowledge Tracing

## Built With

* [MXNet](https://github.com/dmlc/mxnet) - The framework used
* Both Python2 and Python3 are supported

### Prerequisites
* [progress](https://pypi.python.org/pypi/progress) - Dependency package

## Model Architecture

![DKVMN Architecture](https://github.com/jennyzhang0215/DKVMN/blob/master/DKVMN_architecture.png)
![DKVMN Code](https://github.com/jennyzhang0215/DKVMN/blob/master/DKVMN_code.png)

### Data format

The first line the number of exercises a student attempted.
The second line is the exercise tag sequence.
The third line is the response sequence.

 ```
    15
    1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
 ```

### Hyperparameters

--gpus: the gpus will be used, e.g "0,1,2,3"

--max_iter: the number of iterations

--test: enable testing

--train_test: enable testing after training

--show: print progress

--init_std: weight initialization std

--init_lr: initial learning rate

--final_lr: learning rate will not decrease after hitting this threshold

--momentum: momentum rate

--maxgradnorm: maximum gradient norm

--final_fc_dim: hidden state dim for final fc layer

--n_question: the number of unique questions in the dataset

--seqlen: the allowed maximum length of a sequence

--data_dir: data directory

--data_name: data set name

--load: model file to load

--save: path to save model



### Training
 ```
 python main.py --gpus 0
 ```

On Duolingo dataset
```

cd code/python3/

1. python main.py --gpus 0  --n_question 2915 --seqlen 2000 --data_dir ../../data/duolingo --data_name token_unsplitted 
2. python main.py --gpus 0  --n_question 14 --seqlen 2000 --data_dir ../../data/duolingo --data_name part_of_speech_unsplitted 

3. (042418 on PC) python main.py --n_question 3301  --seqlen 10000 --data_dir ../../data/duolingo --data_name new_split_appendFalse_token_unsplitted
4. (042418 on GPU) python main.py --gpus 1 --n_question 3301  --seqlen 10000 --data_dir ../../data/duolingo --data_name new_split_appendFalse_token_unsplitted
5. (042418.toy on CPU/PC) python main.py --n_question 3301  --seqlen 10000 --data_dir ../../data/duolingo --data_name new_split_appendTrue_token_unsplitted_toy --split --batch_size 3 

```
### Testing
 ```
 python main.py --gpus 0 --test True
 ```

## Reference Paper

Jiani Zhang, Xingjian Shi, Irwin King, Dit-Yan Yeung. [Dynamic Key-Value Memory Networks for Knowledge Tracing](https://arxiv.org/pdf/1611.08108.pdf).
In the 26th International Conference on World Wide Web, 2017.

