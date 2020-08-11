import numpy as np
import gc
from ..model_selection import ShuffleSplit
from ..tree import DecisionTreeClassifier
import multiprocessing
from concurrent import futures
from ..tree import tree_prune


def data_scale(input_array,coefficient=0.5):
    max_value = np.max(input_array)
    min_value = np.min(input_array)

    if max_value > min_value:
        scale_array = np.log([((each_value - min_value) / (max_value - min_value) * coefficient) + 1
                            for each_value in input_array])
    else:
        scale_array = 0

    return scale_array

def gini_impurity(a=0,b=0):
    return 1 - np.square(a/(a+b)) - np.square(b/(a+b))


def prune_path(X, y,sample_weight=None,ccp_alphas=None,
                        n_split=5,
                          test_size=0.1, random_state=None,multi_process=False):
    """Cross validation of scores for different values of the decision tree.

    This function allows to test what the optimal size of the post-pruned
    decision tree should be. It computes cross validated scores for different
    size of the tree.

    Parameters
    ----------
    X: array-like of shape at least 2D
        The data to fit.

    y: array-like
        The target variable to try to predict.


    n_iterations : int, optional (default=10)
        Number of re-shuffling & splitting iterations.

    test_size : float (default=0.1) or int
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Returns
    -------
    scores : list of list of floats
        The scores of the computed cross validated trees grouped by tree size.
        scores[0] correspond to the values of trees of size max_n_leaves and
        scores[-1] to the tree with just two leaves.

    """
    kf = ShuffleSplit(n_splits=n_split, test_size=test_size, random_state=10)
    '''Test minimum number of estimator leaves'''


    if multi_process:
        split_index = [(train_index,test_index) for train_index,test_index in kf.split(X,y)]

        Multi_cpu_class = Multicpu( cpu_num=n_split, thread_num=1
                                    ,ccp_alphas=ccp_alphas,random_state=random_state
                                    ,X=X,y=y,sample_weight=sample_weight)
        scores = multi_cpu(multi_cpu_train,split_index,Multi_cpu_class)

        del kf
        gc.collect()
        return scores
    else:

        scores=[]
        for train_index,test_index in kf.split(X,y):
            row_scores = []
            for ccp_alpha in ccp_alphas:
                estimator = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)

                estimator.fit(X[train_index], y[train_index]
                              , sample_weight=sample_weight[train_index])
                row_scores.append(estimator.score(X[test_index], y[test_index]
                                                  , sample_weight=sample_weight[test_index]))
                del estimator


            scores.append(row_scores)
            del row_scores
            gc.collect()

        del kf
        gc.collect()
        return scores



def multi_cpu_train(args,ccp_alphas,random_state,X,y,sample_weight):
    train_index,test_index=args
    estimator = tree_prune.DecisionTreeClassifier(random_state=random_state)
    estimator.fit(X[train_index],y[train_index]
               ,sample_weight=sample_weight[train_index])
    row_scores = []

    # int(0.3*ccp_alphas)
    for i in ccp_alphas:
    #We loop from the bigger values to the smaller ones in order to be
            #able to compute the original tree once, and then make it smaller

            estimator.prune(n_leaves=i)
            if sample_weight is not None:
                row_scores.append(estimator.score(X[test_index], y[test_index],sample_weight=sample_weight[test_index]))
            else:
                row_scores.append(estimator.score(X[test_index], y[test_index]))

    del estimator
    return row_scores


    # row_scores = []
    # for ccp_alpha in ccp_alphas:
    #     estimator = DecisionTreeClassifier(random_state=random_state,ccp_alpha=ccp_alpha)
    #
    #     estimator.fit(X[train_index],y[train_index]
    #            ,sample_weight=sample_weight[train_index])
    #     row_scores.append(estimator.score(X[test_index],y[test_index]
    #                            ,sample_weight=sample_weight[test_index]))
    #
    # del estimator

    #return row_scores




class Multicpu():

    def __init__(self, cpu_num, thread_num,ccp_alphas,random_state,X,y,sample_weight):
        self.cpu_num = cpu_num
        self.thread_num = thread_num
        self.ccp_alphas = ccp_alphas
        self.random_state = random_state
        self.X=X
        self.y=y
        self.sample_weight=sample_weight

    def _multi_cpu(self, func, job_queue, timeout):
        if getLen(job_queue) == 0:
            return []
        index = get_index(job_queue, self.cpu_num)

        cpu_pool = multiprocessing.Pool(processes=self.cpu_num,maxtasksperchild=2)

        mgr = multiprocessing.Manager()
        process_bar = mgr.list()
        for i in range(self.cpu_num):
            process_bar.append(0)

        #print(index)

        result_queue = cpu_pool.map(_multi_thread, [
            [func, self.cpu_num, self.thread_num, job_queue[int(index[i][0]): int(index[i][1]) + 1], timeout, process_bar, i,
              self.ccp_alphas, self.random_state, self.X, self.y, self.sample_weight
             ] for i in range(len(index))])

        cpu_pool.close()
        cpu_pool.join()
        result = []
        for rl in result_queue:
            for r in rl:
                result.append(r)
        return result


def _func(argv):
    #argv[2][argv[3]] = round((argv[4] * 100.0 / argv[5]), 2)
    #sys.stdout.write(str(argv[2]) + ' ||' + '->' + "\r")
    #sys.stdout.flush()
    return argv[0](argv[1],argv[2],argv[3],argv[4],argv[5],argv[6])


def _multi_thread(argv):
    thread_num = argv[2]
    if getLen(argv[3]) < thread_num:
        thread_num = argv[3]
    # func, job_queue, processbar,
    func_argvs = [[argv[0], argv[3][i],argv[7],argv[8],argv[9],argv[10],argv[11]] for i in range(len(argv[3]))]

    result = []
    if thread_num == 1:
        for func_argv in func_argvs:
                        result.append(_func(func_argv))

        return result

    # else
    thread_pool = futures.ThreadPoolExecutor(max_workers=thread_num)

    result = thread_pool.map(_func, func_argvs, timeout=argv[4])



    return [r for r in result]


def get_index(job_queue, split_num):
    job_num = getLen(job_queue)

    if job_num < split_num:
        split_num = job_num
    each_num = job_num / split_num

    index = [[i * each_num, i * each_num + each_num - 1] for i in range(split_num)]

    residual_num = job_num % split_num
    for i in range(residual_num):
        index[split_num - residual_num + i][0] += i
        index[split_num - residual_num + i][1] += i + 1

    return index


def getLen(_list):
    if _list == None:
        return 0
    return len(_list)


def multi_cpu(func, job_queue, Multi_cpu_class, timeout=None):

    return Multi_cpu_class._multi_cpu(func, job_queue, timeout)



