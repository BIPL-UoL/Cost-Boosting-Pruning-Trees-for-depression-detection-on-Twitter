# Cost-sensitive Boosting Pruning Trees for depression detection on Twitter

The Cost-senstive boosting pruning trees code should be compiled with scikit-learn 0.18.2  

""" 

@authors: Lei Tong

@Email: lt228@leicester.ac.uk (Software Developers)

""" 

 

Usage 
======= 

 

Step 1: Download the source code of scikit-learn 0.18.2. 

 

Step 2: Copy the files _tree_prune.cpython-36m-x86_64-linux-gnu.so and tree_prune.py to the folder of scikit-learn-0.18.X/sklearn/tree/. 

 

Step 3: Copy the files boost_modify.py and boost_utils.py to the folder of scikit-learn-0.18.X/sklearn/ensemble/, and add 'from . import boost_utils' and 'from .boost_modify import CBPT' to ensemble/__init__.py file. 

 

Step 4: CD to the root folder of sckikt-learn-0.18.X and execute "pip install -e ." 

 

Testing 
======= 

 

Open an IPython shell:: 

 

	>>> from sklearn.datasets import make_classification 
	>>> from sklearn.ensemble import CBPT 

	>>> X,y = make_classification(n_samples=1000)
	>>> CBPT = CBPT(n_estimators=100,penalty_coef=0.5,penalty_low_limit=1,learning_rate=0.5) 
	>>> CBPT.fit(X,y) 
	>>> CBPT.score(X,y)

 

 

Twitter Depression Detection Datasets 
======= 

Tsinghua Twitter Depression Dataset: http://depressiondetection.droppages.com/

CLPsych 2015 Twitter Dataset: http://www.cs.jhu.edu/ mdredze/clpsych-2015-shared-task-evaluation/

 
