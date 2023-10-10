#generation of functions with known coefficients for tests

import numpy as np
from scipy.stats import multivariate_normal as mvn
import pandas as pd


def generate_function_coefficients(func_specs,n=1):
    """

    sample function coeffieients from normal distribution
    """

    coeffs=np.zeros((n,len(func_specs)))
    for _ in range(n):
        for i in range(len(func_specs)):

            coeffs[_,i] = np.random.normal(loc=func_specs[i][0],scale=func_specs[i][1])
            
    return coeffs

def generate_func_output(func_coeffs, data, intercept):

    wo_noise = np.zeros(len(data))
    for i in range(len(data)):

        coeffs = func_coeffs[i%len(func_coeffs)]
        wo_noise[i]= (data[i]@coeffs) + intercept

    return wo_noise


def generate_correlated_samples(means,n,cov_mat):

    return mvn.rvs(mean = means, cov = cov_mat, size = n)


def bake_off_func_output(train, test, models, error_funcs):

    res = dict()

    for i in models:

        out=list()
        i.fit(train[:,:-1],train[:,-1])

        for j in error_funcs:
                  
            out.append(j(i.predict(test[:,:-1]).squeeze(), test[:,-1].squeeze()))
        
        res[str(i)]=out
        
    return res


def piecewise(input, funcs_dict, critical_vars, default_coeffs=None):
    """
    len of coeffs must be equal to total entries in critical points +1 for each critical var
    coeffs(funciton coefficients )
    critical_points- key=critical_var, val= array

    """

    keys = list(critical_vars.keys())

    #create dictionary to track how often each piece is used
    utilization_dict= dict()
    utilization_dict['default']=0
    for i in funcs_dict.keys():
        utilization_dict[i]=0
    
    y=np.zeros(len(input))
    for x in range(len(input)):

        key=np.zeros(len(keys))
        for i in range(len(key)):

            #get the bucket for this x array by critical variable
            pos = bi(critical_vars[keys[i]], input[x,i])
            key[i]=pos

        #get the proper coefficients for this input array
        key=tuple(key)
        if key in funcs_dict.keys():

            utilization_dict[key]+=1
            y[x]= input[x]@funcs_dict[key]
        else:
            utilization_dict['default']+=1
            y[x]= input[x]@ default_coeffs

    return y, utilization_dict

def weighted_coefs_from_util(funcs_dict, util_dict, defaults):

    avg = np.zeros(len(defaults))
    funcs_dict['default']=np.array(defaults)
    tot = 0
    for i in util_dict:

        avg += util_dict[i]*np.array(funcs_dict[i])
        tot+= util_dict[i]
    
    return avg/tot