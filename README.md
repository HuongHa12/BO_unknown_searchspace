# Bayesian optimization with unknown search space

This repository contains the code for the method GPUCB-UBO proposed in the paper 'Bayesian Optimization with Unknown Search Space', NeurIPS'2019, Ha et al. GPUCB-UBO is a systematic volume expansion
strategy for the Bayesian optimization when the search space is unknown. It guarantees that in iterative expansions of the search space, our method can find a point whose function
value within epsilon of the objective function maximum. Without the need to specify any parameters, GPUCB-UBO automatically triggers a minimal expansion required
iteratively. The method is evaluated on five benchmark test functions and three common machine learning hyper-parameter tuning tasks.

## Prerequisites

- Python 3 (tested with Python 3.6.x)
- Tensorflow (tested with tensorflow 1.10.0)

## Installation

GPUCB-UBO can be directly executed through source code

1. Download and install Python 3.6.x [here](https://www.python.org/downloads/).

2. Install Tensorflow (if testing with the machine learning models)

    ```$ pip install tensorflow==1.10.0```

3. Clone GPUCB-UBO

    ``` $ clone https://github.com/HuongHa12/BO_unknown_searchspace.git```


## Test Cases

GPUCB-UBO has been evaluated on 7 synthetic benchmark functions: Beale, Eggholder, Hartman3, Levy3, Hartman6, Levy10, Ackley10 and three common machine learning hyper-parameter tuning tasks: linear regression with elastic net, multilayer perceptron
and convolutional neural network. The codes to implement these synthetic functions and ML models can be found in the scripts ```functions.py``` and ```functions_ml_model.py```.

## Usage
To run GPUCB-UBO, users need to specify the name of the test case they wish to evaluate and then run the scripts ```BO_unknown_searchspace.py``` (for synthetic functions) or ```BO_unknown_searchspace_MLmodels.py``` (for machine learning models). The name of the test case can be specified in Line 23 or 24 of the two scripts.

## Citing GPUCB-UBO
If you find our code useful, please kindly cite our paper. 

```
@inproceedings{Ha2019,
  title={Bayesian Optimization with Unknown Search Space},
  author={Ha, Huong and Rana, Santu and Gupta, Sunil and Nguyen, Thanh and Tran-The, Hung and Venkatesh, Svetha},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

## License
GPUCB-UBO is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
