# An Exploration into Evolution Strategies



Accompanying write-up can be found on my [medium post](https://medium.com/@jovansardinha/an-exploration-into-evolution-strategies-97c42122c486)

**objectives**:  
1 — Implement evolution strategies from scratch and use it to optimize the weights of a neural network on the task of MNIST digit recognition.  
2 — Find a good set of hyperparameters of the algorithm that achieve the best results after 12 hours of training.  
3 — Distribute the above across the cores of a computer (going to 4 cores). Analyze the speedup observed when going from 1 core to a 4 core implementation.  


**Folder structure**:

```
.
├── evolution/
|   ├── tests/  # contains all test    
|   ├── __inti__.py									
|   ├── es.py	  # implimentation of ES class  
|   └── main.py  # main file to execute   
├── notebooks/
|   ├── Analyzing Results of Best Hyperparameter.ipynb  # analyzing best parameter results  
|   ├── Analyzing Results of Hyperparameter Search.ipynb # post analysis for hyperparameter tuning  
|   ├── Analyzing Times of Runs by Number of Workers.ipynb # post analysis of runtimes  
|   ├── MNIST - Keras.ipynb		 # MNIST keras models  
|   └── bare bones implementation of NES - karpathy.ipynb # karpathy ES starter  
|── .gitignore
|── .README.md
└── requirements.txt  # list of packages used
```


**Built and tested on**:  
operating system: Ubuntu 16.04.2 LTS  
python version: 3.5.2  
pip version: 9.0.1
