# Cost-sensitive Boosting Pruning Trees for depression detection on Twitter

The Cost-senstive boosting pruning trees code should be compiled with scikit-learn 0.18.2  

""" 

@authors: Lei Tong

@Email: lt228@leicester.ac.uk

""" 

This method is built upon [sklearn_tree_post_prune](https://github.com/shenwanxiang/sklearn-post-prune-tree/tree/master) and scikit-learn 0.18.X.

Usage 
======= 

Step 0: Configure the Python Environment
  ```
  conda create -n py36 python=3.6.8
  ```
Step 1: Install Required Dependencies
  ```
  source activate py36
  pip install -r requirements.txt
  ```
Step 2: Download the source code of [scikit-learn 0.18.0](https://github.com/scikit-learn/scikit-learn/tree/0.18.X) and extract it. 

Step 3: Build Cython Extensions
  ```
  cd ./sklearn_tree_post_prune/src/
  cython _tree_prune.pyx
  python setup.py build
  ```
Step 4: Copy the following files to the directory scikit-learn-0.18.X/sklearn/tree/
  ```
  cp ./sklearn_tree_post_prune/src/build/lib_Completement with the system version/tree/_tree_prune.cpython-36m-darwin.so to scikit-learn-0.18.X/sklearn/tree/
  cp ./tree_prune.py to scikit-learn-0.18.X/sklearn/tree/
  ```

Step 5: Copy the following files to the directory scikit-learn-0.18.X/sklearn/ensemble/ and add 'from . import boost\_utils' to ensemble/\_\_init\_\_.py file:
  ```
  cp ./boost_modify.py to scikit-learn-0.18.X/sklearn/ensemble/
  cp ./boost_utils.py to scikit-learn-0.18.X/sklearn/ensemble/
  ```
 
Step 6: Change the directory to the root folder of scikit-learn-0.18.X and execute the following command:
 ```
  cd ./scikit-learn-0.18.X/
  pip install -e .
  ```


Testing 
======= 

 

Open an IPython shell:: 

 

	>>> from sklearn.datasets import make_classification 
	>>> from sklearn.ensemble import CBPT 

	>>> X,y = make_classification(n_samples=1000)
	>>> clf = CBPT(n_estimators=100,penalty_coef=0.5,penalty_low_limit=1,learning_rate=0.5) 
	>>> clf.fit(X,y) 
	>>> clf.score(X,y)

 

Twitter Depression Detection Datasets 
======= 

Tsinghua Twitter Depression Dataset: http://depressiondetection.droppages.com/

CLPsych 2015 Twitter Dataset: http://www.cs.jhu.edu/mdredze/clpsych-2015-shared-task-evaluation/


## Acknowledgement
Many thanks to the authors of [sklearn_tree_post_prune](https://github.com/shenwanxiang/sklearn-post-prune-tree/tree/master). 


Citation 
======= 
If you find CBPT useful in your research, please consider citing:
	
	@article{tong2022cost,
	  title={Cost-sensitive Boosting Pruning Trees for depression detection on Twitter},
	  author={Tong, Lei and Liu, Zhihua and Jiang, Zheheng and Zhou, Feixiang and Chen, Long and Lyu, Jialin and Zhang, Xiangrong and Zhang, Qianni and Sadka, 	     Abdul and Wang, Yinhai and others},
	  journal={IEEE Transactions on Affective Computing},
	  year={2022},
	  publisher={IEEE}
	}

