#evaluation of models and their corresponding estimators

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def rmse(pred, true):

    return np.sqrt(sum((true-pred)**2)/len(pred))



def rank_by_coef_error(models, weighted_coefs):
    """ 
    higher is better
    """

    res=dict()
    for key, val in models.items():

        res[key] = cosine_similarity(val.reshape(1,-1), weighted_coefs.reshape(1,-1))
    
    return res