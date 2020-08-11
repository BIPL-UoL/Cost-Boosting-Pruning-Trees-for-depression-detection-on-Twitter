"""Weight Boosting

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- ``AdaBoostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.

- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  regression problems.
"""

# Authors: Noel Dawe <noel@dawe.me>
#          Gilles Louppe <g.louppe@gmail.com>
#          Hamzeh Alsalhi <ha258@cornell.edu>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.core.umath_tests import inner1d

from .base import BaseEnsemble
from ..base import ClassifierMixin, RegressorMixin, is_regressor
from ..externals import six
from ..externals.six.moves import zip
from ..externals.six.moves import xrange as range
from .forest import BaseForest
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..tree.tree import BaseDecisionTree
from ..tree._tree import DTYPE
from ..utils import check_array, check_X_y, check_random_state
from ..metrics import accuracy_score, r2_score
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from ..tree import tree_prune
from copy import deepcopy
from ..ensemble import boost_utils

__all__ = [
    'CBPT',

]


class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None):

        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y,v_Folds, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))


        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)



        from tqdm import tqdm
        for iboost in tqdm(range(self.n_estimators),ascii=True,desc='Finished trees'):
            # Boosting step

            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state,v_Folds)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                print('estimator error is zero!')
                pass

            #print(sample_weight)
            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y,sample_weight, random_state,v_Folds):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        random_state : numpy.RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like, shape = [n_samples]
            Labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        z : float
        """
        for y_pred in self.staged_predict(X):
            if isinstance(self, ClassifierMixin):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                              * log_proba.sum(axis=1)[:, np.newaxis])


class CBPT(BaseWeightBoosting, ClassifierMixin):
    """An CBPT classifier.



    This class implements the algorithm known as AdaBoost-SAMME [2].

    Read more in the :ref:`User Guide <adaboost>`.

    Parameters
    ----------
    base_estimator : object, optional (default=tree_prune.DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.

    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    penalty_coef : float (default=0.5)
        Data scaling coefficient for penalty terms.

    penalty_low_limit: float (default=1)
        Data scaling low limit for penalty terms.

    pruning_stride: integer (default=1),
        Control the stride for pruning leaves searching, s.t. stride >= 1 & stride < max_leaves

    pruning_start:  float (default=0.9),
        Control the starting percentage for pruning leaves searching

    pruning_end:  float (default=0.1),
        Control the ending percentage for pruning leaves searching

    algorithm : {'SAMME', }, optional
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.



    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME',
                 penalty_coef = 0.5,
                 penalty_low_limit = 1,
                 penalty_normalize=False,
                 pruning_stride =1,
                 pruning_start=0.9,
                 pruning_end=0.1,
                 random_state=None):

        super(CBPT, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
        # Boolean for penalty term
        self.penalty_coef  = penalty_coef
        self.penalty_low_limit = penalty_low_limit
        self.penalty_normalize=penalty_normalize
        self.pruning_stride = pruning_stride
        self.pruning_start = pruning_start
        self.pruning_end = pruning_end

    def fit(self, X, y ,v_Folds=5, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        v_Folds : Integer (default=5)
            Shuffle Split for the training set
            ShuffleSplit(n_splits=5, test_size=0.1, random_state=10)

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super(CBPT, self).fit(X, y,v_Folds, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(CBPT, self)._validate_estimator(
            default=tree_prune.DecisionTreeClassifier())

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _boost(self, iboost, X, y,sample_weight, random_state,v_Folds):
        """Implement a single boost.

        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        random_state : numpy.RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        if self.algorithm == 'SAMME':
            return self._boost_discrete(iboost, X, y, sample_weight,
                                        random_state,v_Folds)





    def _boost_discrete(self, iboost, X, y, sample_weight, random_state,v_Folds):
        """Implement a single boost using the SAMME discrete algorithm."""
        from ..tree import tree_prune
        from copy import deepcopy
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)


        scores,leaves_number = tree_prune.prune_path_multiprocess(estimator,
                                       X, y, n_iterations=v_Folds, sample_weight=sample_weight
                                    ,pruning_stride=self.pruning_stride,pruning_start=self.pruning_start
                                                                  ,pruning_end=self.pruning_end)
        scores = list(scores)

        estimator_copy = deepcopy(estimator)


        #print('sample weight:',sample_weight)
        means = np.array([np.mean(s) for s in scores])
        assert means.shape == leaves_number.shape
        if means.shape[0] ==0:
            pass
        else:
            max_means_leaf = leaves_number[np.argmax(means)]
            estimator_leaves = tree_prune.get_n_leaves(estimator)
            #print('prune_leaf:%d, estimator_leaf:%f'%(max_means_leaf,estimator_leaves))
            if estimator_leaves <= max_means_leaf:
                estimator.prune(estimator_leaves-1)
            else:
                estimator.prune(max_means_leaf)



        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction

        estimator_error = np.mean(
             np.average(incorrect, weights=sample_weight, axis=0))

        # Boosting penalty term

        depth_term = 0
        gini_term = 0


        if self.penalty_coef != None or self.penalty_low_limit != None:

            nodes_id,nodes_depth,nodes_values,nodes_samples = tree_prune.get_sample_depth(estimator,X)

            test_value_range= nodes_values<1
            assert test_value_range.all()==True

            gini_term_coef = True
            depth_term_coef = True
            if gini_term_coef!=None:

                #recompute gini index without consider sample weight

                y_copy = y.astype('int64')
                unique_label = np.unique(y_copy)
                max_label = max(unique_label)
                calculated_gini = np.zeros(max(nodes_id)+1)
                for unique_ids in np.unique(nodes_id):

                    find_nodes = np.where(nodes_id == unique_ids)
                    pick_y_train = y_copy[find_nodes]

                    #pick_y_train = pick_y_train.astype('int64')
                    dist_count = np.bincount(pick_y_train,minlength=max_label+1)

                    node_sample_vector = dist_count[unique_label]

                    # Calcualte Gini index
                    sum_node= sum(node_sample_vector)
                    max_node = max(node_sample_vector)
                    calculated_gini[unique_ids] = boost_utils.gini_impurity(max_node,sum_node-max_node)


                sample_number = X.shape[0]

                #original_impurity = estimator.tree_.init_error[0]
                assert max(calculated_gini) <= 0.5

                gini_gains = np.abs([((1-nodes_samples[index]/sample_number) * (0.5-calculated_gini[nodes_id[index]]))
                                     for index in range(len(nodes_id))])

                # gini_gains = np.abs([nodes_values[index]-original_impurity
                #                      for index in range(len(nodes_id))])
                #gini_gains = original_impurity / gini_gains

                #gini_term = np.log(gini_gains)
                # Data scale:
                #gini_gains = [max(gini_gains)-gain_value for gain_value in gini_gains]

                max_gains = np.max(gini_gains)
                min_gains = np.min(gini_gains)

                if max_gains>min_gains:
                    gini_term = np.log([((gain_value- min_gains) / (max_gains - min_gains)*self.penalty_coef) + self.penalty_low_limit
                                             for gain_value in gini_gains])
                else:
                    gini_term = 0

            if depth_term_coef !=None:

                #nodes_depth = [max(nodes_depth)-depth_value for depth_value in nodes_depth]
                max_depth = np.max(nodes_depth)
                min_depth = np.min(nodes_depth)


                if max_depth > min_depth:
                    depth_term = np.log([((depth_value - min_depth) / (max_depth - min_depth)*self.penalty_coef) + self.penalty_low_limit
                                         for depth_value in nodes_depth])
                else:
                    depth_term = 0

            # if self.depth_term_coef !=None and self.gini_term_coef!=None and self.penalty_normalize == True:
            #     #normalize penality term range to (1,2)
            #     depth_term = boost_utils.data_scale(input_array=np.exp(depth_term+gini_term),coefficient=1)
            #     gini_term=0


        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing



        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            self.estimators_.append(estimator_copy)
            '''Reset sample weight'''
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
            print('Reset sample weight')
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return sample_weight, 1., estimator_error





        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (

            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again

        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp((estimator_weight + depth_term + gini_term) * incorrect
                                    *((sample_weight > 0) |
                                     (estimator_weight < 0)))





        return sample_weight, estimator_weight, estimator_error



    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """


        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(_samme_proba(estimator, n_classes, X)
                       for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = self._validate_X_predict(X)

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(_samme_proba(estimator, n_classes, X)
                        for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        proba = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_proba = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_proba = estimator.predict_proba(X) * weight

            if proba is None:
                proba = current_proba
            else:
                proba += current_proba

            real_proba = np.exp((1. / (n_classes - 1)) * (proba / norm))
            normalizer = real_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            real_proba /= normalizer

            yield real_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))


