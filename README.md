# Cost-sensitive Boosting Pruning Trees for depression detection on Twitter

The Inverse boosting pruning trees code should be compiled with scikit-learn 0.18.0   

(This code will be updated later and be compatible with new version of sklearn) 

""" 

@authors: Lei Tong

@Email: lt228@leicester.ac.uk (Software Developers)

""" 

 

Usage 
======= 

 

Step 1: Download the source code of scikit-learn 0.18.0. 

 

Step 2: Copy the files _tree_prune.cpython-35m-x86_64-linux-gnu.so and tree_prune.py to the folder of scikit-learn-0.18.X/sklearn/tree/. 

 

Step 3: Copy the files boost_modify.py and boost_utils.py to the folder of scikit-learn-0.18.X/sklearn/ensemble/, and add 'from . import boost_utils' to ensemble/__init__.py file. 

 

Step 4: CD to the root folder of sckikt-learn-0.18.X and execute "pip install -e ." 

 

 

 

Testing 
======= 

 

Open an IPython shell:: 

 

	>>> from sklearn.datasets import make_classification 
	>>> from sklearn.tree import tree_prune 
	>>> from sklearn.ensemble import boost_modify 

	>>> X,y = make_classification(n_samples=20)
	>>> IBPT = boost_modify.AdaBoostClassifier(n_estimators=100,
							base_estimator=tree_prune.DecisionTreeClassifier(),algorithm='SAMME') 
	>>> IBPT.fit(train_data,train_target,v_Folds=5) 
	>>> predicted_results = IBPT.predict(your_testing_data) 

 

 

Undertaking earthquake prediction performance  
======= 

Run the testing example by "python test.py" to have the prediction results file. 

Run the code by "Rscript caculate_metrics.R" to have the earthquake prediction metrics and curves. 

 
