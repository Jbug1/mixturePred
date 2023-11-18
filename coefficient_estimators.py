#funcitons for coefficient estimation
#all functions use sklearn tree structs

import numpy as np

def traverse_to_node(x, tree, feature, feature_means, i=0, prop=1):
        """ 
        tree: decision tree object
        feature: feature that we are holding blank
        i: index at which to begin in attribute arrays
        prop: proportion of weight we are assigning to this prediction
        """

        #return prediction with proper proportion if this is a node
        if tree.children_left[i]==-1:

            return (tree.value[i][0][0], feature_means[i][feature], prop)
            
        #normal traverse if this is not the feature of interest
        if tree.feature[i]!= feature:
             
             if x[tree.feature[i]] <= tree.threshold[i]:
                  
                  return traverse_to_node(x, tree, feature, feature_means, i = tree.children_left[i], prop = prop)
             else:
                  return traverse_to_node(x, tree, feature, feature_means, i = tree.children_right[i], prop = prop)

        #otherwise we need to do proportional split without conditioning on the value of feature
        else:
            
            #aesthetic only
            samps = tree.n_node_samples

            #traverse to left with proper proportions
            prop_left = prop * samps[tree.children_left[i]]/samps[i]
            left = traverse_to_node(x, tree, feature, feature_means, i = tree.children_left[i], prop = prop_left)
            
            #traverse to right with proper proportions
            prop_right = prop * samps[tree.children_right[i]]/samps[i]
            right = traverse_to_node(x, tree, feature, feature_means, i = tree.children_right[i], prop = prop_right)
            
            return (left[0]*prop_left + right[0]*prop_right, left[1]*prop_left + right[1]*prop_right, prop_left+prop_right)
        


def coef_impact_estimate(x, tree, feature, feature_means, pred_val, i=0):
    """
    Estimator of coeff value conditioned on values of other features

    formula (y_pred - y|coeffs_~feature_x)/(feature_x- E[feature_x|features_~feature_x])
    x: np array of one observations
    tree: tree object
    feature: int index of feature we are estimating
    feature_means: means of all features and response by node
    i: int index of current node
    """

    #don't want to include impact on coeff if 0
    #don't do this...we still need to get the conditional mean
    # if x[feature] - feature_means[i][feature]==0:
    #     return None
    
    #get masked_prediction
    masked_pred_val, masked_feature_mean, _ = traverse_to_node(x, tree, feature, feature_means)

    #this means the feature was not used or the split predictions resulted in same feature mean estimate
    if x[feature] == masked_feature_mean:
        return 0

    return (pred_val - masked_pred_val) / (x[feature] - masked_feature_mean)

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

def estimate_all_coefs_ensemble(x, mod, feature_means_forest):

    ests = np.zeros((len(mod.estimators_),len(x)))

    for i in range(len(x)):
        for j in range(len(mod.estimators_)):

            #catch RF and GBM being organized differently. very sloppy
            try:
                pred_val = mod.estimators_[j].predict([x])
                ests[j,i] = coef_impact_estimate(x, mod.estimators_[j].tree_, feature =i, feature_means = feature_means_forest[j], pred_val=pred_val)
            except:
                pred_val = mod.estimators_[j][0].predict([x])
                ests[j,i] = coef_impact_estimate(x, mod.estimators_[j][0].tree_, feature =i, feature_means = feature_means_forest[j], pred_val=pred_val)

    return np.mean(ests, axis=0)

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
    for i in range(len(models)):

        #this is an ensemble model
        if hasattr(models[i], 'estimators_'):

            est = estimate_all_coefs_for_dataset(train[:,:-1], models[i], feature_means[i])
            res[models[i]]=np.nanmean(est,axis=0)

        #this is a decision tree
        elif hasattr(models[i], 'tree_'):

            est = estimate_all_coefs_for_dataset(train[:,:-1], models[i], feature_means[i], ensemble=False)
            res[models[i]]=np.nanmean(est, axis=0)

        #this is a regression model
        else:
            res[i]=i.coef_

    return res

def get_feature_means_models(X, models):

    feature_means=list()

    for model in models:

        feature_means.append(get_feature_means_forest(X, model))

    return  feature_means

def get_feature_means_forest(X,model):

    #catch all architectures

    if hasattr(model, 'tree_'):

        return [get_feature_means_tree(X, model.tree_)]
    
    feature_means=np.zeros(len(model.estimators_), dtype=object)
    for i in range(len(model.estimators_)):

        #catch different organization of rf and gbm
        try:
            feature_means[i] = get_feature_means_tree(X,model.estimators_[i].tree_)
        except:
            feature_means[i] = get_feature_means_tree(X,model.estimators_[i][0].tree_)

    return feature_means

def get_feature_means_tree(X, tree):

    #this will hold how many times we have visited a given leaf
    seens = np.zeros(len(tree.children_left))

    #this will hold means by node
    feature_means = np.zeros((len(tree.children_left), X.shape[1]))

    for i in range(X.shape[0]):

        ind = pred_leaf(X[i], tree)
        
        first_weight = seens[ind]/(seens[ind]+1)
        second_weight = 1/(seens[ind]+1)
        feature_means[ind] = (feature_means[ind]*first_weight) + X[i]*second_weight
        seens[ind]+=1
    
    return feature_means


def pred_leaf(x, tree):
    """
    which node does this array end up at?
    """

    i=0
    #while we are not at a node
    while tree.children_left[i]!=-1:

        if x[tree.feature[i]] <= tree.threshold[i]:
            i = tree.children_left[i]
        else:
            i = tree.children_right[i]

    return i
    



