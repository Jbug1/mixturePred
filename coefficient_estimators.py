#funcitons for coefficient estimation
#all functions use sklearn tree structs

import numpy as np


def weighted_coefs_from_util(funcs_dict, util_dict, defaults):

    avg = np.zeros(len(defaults))
    funcs_dict['default']=np.array(defaults)
    tot = 0
    for i in util_dict:

        avg += util_dict[i]*np.array(funcs_dict[i])
        tot+= util_dict[i]
    
    return avg/tot


def traverse_to_node(x, tree, feature, i=0, prop=1):
        """ 
        tree: decision tree object
        feature: feature that we are holding blank
        i: index at which to begin in attribute arrays
        prop: proportion of weight we are assigning to this prediction
        """

        #return prediction with proper proportion if this is a node
        if tree.children_left[i]==-1:

            return (tree.value[i][0][0], prop)
            
        #normal traverse if this is not the feature of interest
        if tree.feature[i]!= feature:
             
             if x[tree.feature[i]] <= tree.threshold[i]:
                  
                  return traverse_to_node(x, tree, feature, i = tree.children_left[i], prop = prop)
             else:
                  return traverse_to_node(x, tree, feature, i = tree.children_right[i], prop = prop)

        #otherwise we need to do proportional split without conditioning on the value of feature
        else:
            
            #aesthetic only
            samps = tree.n_node_samples

            #traverse to left with proper proportions
            prop_left = prop * samps[tree.children_left[i]]/samps[i]
            left =traverse_to_node(x, tree, feature, i = tree.children_left[i], prop = prop_left)
            
            
            #traverse to right with proper proportions
            prop_right = prop * samps[tree.children_right[i]]/samps[i]
            right = traverse_to_node(x, tree, feature, i = tree.children_right[i], prop = prop_right)
            
            return (left[0]*(left[1]/(left[1]+right[1])) + right[0]*(right[1]/(left[1]+right[1])),left[1]+right[1])
        


def coef_impact_estimate(x, tree, feature, feature_mean, pred_val):

    #don't want to include impact on coeff if 0
    if x[feature] -feature_mean ==0:
        return None
    
    #get masked_prediction
    masked_pred_val = traverse_to_node(x, tree, feature)[0]

    return (pred_val - masked_pred_val) / (x[feature] -feature_mean)

def estimate_all_coefs(x, tree, feature_means):

    #create array to catch coeff estimates
    ests = np.zeros(len(x))

    #get unmasked prediction value
    try:
        pred_val = tree.predict(x.astype('float32').reshape(1,-1))[0]
    except:
        print (x)
    for i in range(len(x)):

        ests[i] = coef_impact_estimate(x, tree, feature=i, feature_mean = feature_means[i], pred_val = pred_val)

    return ests

def estimate_all_coefs_ensemble(x, mod, feature_means):

    ests = np.zeros((len(mod.estimators_),len(x)))

    for i in range(len(x)):
        for j in range(len(mod.estimators_)):

            #catch RF and GBM being organized differently. very sloppy
            try:
                pred_val = mod.estimators_[j].predict([x])
                ests[j,i] = coef_impact_estimate(x, mod.estimators_[j].tree_, feature =i, feature_mean = feature_means[i], pred_val=pred_val)
            except:
                pred_val = mod.estimators_[j][0].predict([x])
                ests[j,i] = coef_impact_estimate(x, mod.estimators_[j][0].tree_, feature =i, feature_mean = feature_means[i], pred_val=pred_val)

    return np.nanmean(ests, axis=0)

def estimate_all_coefs_for_dataset(x, mod, feature_means, ensemble=True):

    res = np.zeros(x.shape)
    for i in range(len(x)):

        if ensemble:
            res[i] = estimate_all_coefs_ensemble(x[i], mod, feature_means)

        else:
            res[i] = estimate_all_coefs(x[i], mod.tree_, feature_means)

    return res



def get_coef_estimates(models, train, feature_means):

    res = dict()
    for i in models:

        #this is an ensemble model
        if hasattr(i, 'estimators_'):

            est = estimate_all_coefs_for_dataset(train[:,:-1], i, feature_means)
            res[i]=np.nanmean(est,axis=0)

        #this is a decision tree
        elif hasattr(i, 'tree_'):

            est = estimate_all_coefs_for_dataset(train[:,:-1],i, feature_means, ensemble=False)
            res[i]=np.nanmean(est, axis=0)

        #this is a regression model
        else:
            res[i]=i.coef_

    return res

#get model estimates of all coefficients
#estimates = get_coef_estimates(models, train, )




