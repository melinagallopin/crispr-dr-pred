# distutils: language=c++
cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& value) except +
import numpy as np
cimport numpy as np
import array as ar
from cpython cimport array as ar
from sklearn.utils._random cimport sample_without_replacement
from sklearn.tree._utils cimport rand_uniform
cimport cython
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport log
ctypedef np.uint8_t uint8
ctypedef np.npy_uint32 UINT32_t
#@cython.wraparound(False)


cdef double c_prob_information_gain(np.ndarray[np.float_t, ndim=2] y, np.ndarray[np.float_t, ndim=1] feature, double cutoff, np.str criterion):
    # left and right are matrices with K columns
    cdef:
        int n_left=0, n_right=0, n_parent=0
        #double [:, :] y_left = y[feature < cutoff, :]
        #double [:, :] y_right = y[feature >= cutoff, :]
        int i, j
        double value_left = 0.0
        double value_right=0.0
        double value=0.0
        double max_prop = - 1.0
        double max_prop_left = - 1.0
        double max_prop_right = - 1.0
        int n = y.shape[0]
        int K = y.shape[1]
        vector[double] props_left = vector[double](K, 0.0)
        vector[double] props_right = vector[double](K, 0.0)
        vector[double] props = vector[double](K, 0.0)
    for i in range(n):
        if feature[i] < cutoff:
            for j in range(K):
                props_left[j] = props_left[j] + y[i, j]
            n_left = n_left + 1
        else:
            for j in range(K):
                props_right[j] = props_right[j] + y[i, j]
            n_right = n_right + 1

    for j in range(K):
        props[j] = (props_right[j] + props_left[j]) / n
        if n_left != 0:
            props_left[j] = props_left[j] / n_left
        if n_right != 0:
            props_right[j] = props_right[j] / n_right
        if criterion == 'gini':
            value = value + props[j] * (1 - props[j])
            value_left = value_left + props_left[j] * (1 - props_left[j])
            value_right = value_right + props_right[j] * (1 - props_right[j])
        elif criterion == 'misclass':
            if props[j] > max_prop:
                max_prop = props[j]
                value = 1 - max_prop
            if props_left[j] > max_prop_left:
                max_prop_left = props_left[j]
                value_left = 1 - max_prop_left
            if props_right[j] > max_prop_right:
                max_prop_right = props_right[j]
                value_right = 1 - max_prop_right
        # criterion == 'entropy'
        else :
            if props[j] != 0.0:
                value = value - props[j] * log(props[j])
            if props_left[j] != 0.0:
                value_left = value_left - props_left[j] * log(props_left[j])
            if props_right[j] != 0.0:
                value_right = value_right - props_right[j] * log(props_right[j])
    return value - (n_left * value_left + n_right * value_right) / n


cpdef (int, double, double) c_find_best_split_of_all(np.ndarray[np.float_t, ndim=2] x, np.ndarray[np.float_t, ndim=2] y,
                                                    np.str criterion, int max_features, np.str splitter, int num_rand_splits, UINT32_t random_state):
    cdef:
        np.ndarray[np.int_t, ndim=1] feat_inds = sample_without_replacement(x.shape[1], max_features, 'auto')
        np.ndarray[np.float_t, ndim=2] best_feat_splits = np.zeros((feat_inds.shape[0], 2), dtype=np.double)
        int i, feat_idx
        int idx_max
        double max_gain = - 1.0
    for i in range(feat_inds.shape[0]):
        feat_idx = feat_inds[i]
        best_feat_splits[i, 0], best_feat_splits[i, 1] = c_find_best_split(x[:, feat_idx], y, criterion, splitter, num_rand_splits, random_state)
        # maximum search
        if best_feat_splits[i, 0] > max_gain:
            max_gain = best_feat_splits[i, 0]
            idx_max = i
    cdef:
        double cutoff = best_feat_splits[idx_max, 1]
        int chosen_feat_idx = feat_inds[idx_max]
    return chosen_feat_idx, cutoff, max_gain

cdef (double, double) c_find_best_split(np.ndarray[np.float_t, ndim=1] feature, np.ndarray[np.float_t, ndim=2] y, np.str criterion,
                                        np.str splitter, int num_rand_splits, UINT32_t random_state):
    cdef:
        # np.ndarray[np.float_t, ndim=1] feature_values = np.unique(feature)
        # np.ndarray[np.float_t, ndim=1] gains
        vector[double] values, unique_values
        double feature_min, feature_max
        int i, idx_max
        double value, max_gain, cutoff, gain
    feature_min = np.double(np.iinfo(np.int32).max)
    feature_max = - feature_min
    for i in range(feature.shape[0]):
        # if feature[i] is not unique w.r.t. unique_values
        if find(unique_values.begin(), unique_values.end(), feature[i]) == unique_values.end():
            unique_values.push_back(feature[i])
            if feature[i] < feature_min:
                feature_min = feature[i]
            if feature[i] > feature_max:
                feature_max = feature[i]
    max_gain = - 1.0
    # a set of possible cutoffs
    if splitter == 'random':
        for i in range(num_rand_splits):
            values.push_back(rand_uniform(feature_min, feature_max, &random_state))
    # best splitter
    else:
        values = unique_values
    # evaluate information gain
    for i in range(values.size()):
        #if values[i] <= feature_min or values[i] >= feature_max:
        gain = c_prob_information_gain(y, feature, values[i], criterion)
        # find cutoff that maximizes information gain
        if gain > max_gain:
            idx_max = i
            max_gain = gain
    cutoff = values[idx_max]
    return max_gain, cutoff


cpdef c_get_prediction(np.ndarray[np.float_t, ndim=2] x, list tree, int K):
    cdef:
        dict node
        bool cond
        np.ndarray[np.float_t, ndim=2] probs = np.empty((x.shape[0], K), dtype=np.double)

    for i in range(x.shape[0]):
        cond = True
        node = tree[0]
        while cond:
            if node['feat_idx'] is None:
                probs[i, :] = node['props']
                cond = False
            elif x[i, node['feat_idx']] < node['cutoff']:
                node = tree[node['left']]
            else:
                node = tree[node['right']]
    return probs


# semi-supervised
cdef double c_variance(np.ndarray[np.float_t, ndim=1] feature, double cutoff):
    # left and right are matrices with K columns
    cdef:
        int n_left=0, n_right=0
        int i
        double sum_of_squares=0.0
        double mean = 0.0
        double mean_left = 0.0
        double mean_right = 0.0
        int n = feature.shape[0]

    for i in range(n):
        if feature[i] < cutoff:
            n_left = n_left + 1
            mean_left = mean_left + feature[i]
        else:
            n_right = n_right + 1
            mean_right = mean_right + feature[i]
        mean = mean + feature[i]
    if n_left == 0 or n_right == 0:
        return 0
    mean = mean / n
    mean_left = mean_left / n_left
    mean_right = mean_right / n_right
    for i in range(n):
        sum_of_squares = sum_of_squares + (feature[i] - mean) * (feature[i] - mean)
    return (n_left * (mean_left - mean) * (mean_left - mean) + n_right * (mean_right - mean) * (mean_right - mean)) / sum_of_squares


cpdef (int, double, double) semi_c_find_best_split_of_all(np.ndarray[np.float_t, ndim=2] x, np.ndarray[np.float_t, ndim=2] x_l,
                                                    np.ndarray[np.float_t, ndim=2] y_l, double lam,
                                                    np.str criterion, int max_features, np.str splitter, int num_rand_splits, UINT32_t random_state):
    cdef:
        np.ndarray[np.int_t, ndim=1] feat_inds = sample_without_replacement(x.shape[1], max_features, 'auto')
        np.ndarray[np.float_t, ndim=2] best_feat_splits = np.zeros((feat_inds.shape[0], 2), dtype=np.double)
        int i, feat_idx
        int idx_max
        double max_gain = - 1.0
    for i in range(feat_inds.shape[0]):
        feat_idx = feat_inds[i]
        best_feat_splits[i, 0], best_feat_splits[i, 1] = semi_c_find_best_split(x[:, feat_idx], x_l[:, feat_idx], y_l, lam, criterion, splitter, num_rand_splits, random_state)
        # maximum search
        if best_feat_splits[i, 0] > max_gain:
            max_gain = best_feat_splits[i, 0]
            idx_max = i
    cdef:
        double cutoff = best_feat_splits[idx_max, 1]
        int chosen_feat_idx = feat_inds[idx_max]
    return chosen_feat_idx, cutoff, max_gain

cdef (double, double) semi_c_find_best_split(np.ndarray[np.float_t, ndim=1] feature, np.ndarray[np.float_t, ndim=1] sup_feature, np.ndarray[np.float_t, ndim=2] y, double lam, np.str criterion,
                                        np.str splitter, int num_rand_splits, UINT32_t random_state):
    cdef:
        # np.ndarray[np.float_t, ndim=1] feature_values = np.unique(feature)
        # np.ndarray[np.float_t, ndim=1] gains
        vector[double] values, unique_values
        double feature_min, feature_max
        int i, idx_max
        double value, max_gain, cutoff, gain, unsup_gain, sup_gain
    feature_min = np.double(np.iinfo(np.int32).max)
    feature_max = - feature_min
    if lam == 0.0:
        for i in range(feature.shape[0]):
            # if feature[i] is not unique w.r.t. unique_values
            if find(unique_values.begin(), unique_values.end(), feature[i]) == unique_values.end():
                unique_values.push_back(feature[i])
                if feature[i] < feature_min:
                    feature_min = feature[i]
                if feature[i] > feature_max:
                    feature_max = feature[i]
    else:
        for i in range(sup_feature.shape[0]):
            # if sup_feature[i] is not unique w.r.t. unique_values
            if find(unique_values.begin(), unique_values.end(), sup_feature[i]) == unique_values.end():
                unique_values.push_back(sup_feature[i])
                if sup_feature[i] < feature_min:
                    feature_min = sup_feature[i]
                if sup_feature[i] > feature_max:
                    feature_max = sup_feature[i]

    max_gain = - 1.0
    # a set of possible cutoffs
    if splitter == 'random':
        for i in range(num_rand_splits):
            values.push_back(rand_uniform(feature_min, feature_max, &random_state))
    # best splitter
    else:
        values = unique_values
    # evaluate information gain
    for i in range(values.size()):
        if lam == 1.0 or y[0, 0] == -1.0:
            sup_gain = 0.0
        else:
            sup_gain = c_prob_information_gain(y, sup_feature, values[i], criterion)
        if lam == 0.0:
            unsup_gain = 0.0
        else:
            unsup_gain = c_variance(feature, values[i])
        gain = (1 - lam) * sup_gain + lam * unsup_gain
        # find cutoff that maximizes information gain
        if gain > max_gain:
            idx_max = i
            max_gain = gain
    cutoff = values[idx_max]
    return max_gain, cutoff
