#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:19:51 2024

@author: dliu
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv(
    "./data/pm25/SampleData/pm25_ground.txt",
    index_col="datetime",
    parse_dates=True,
)


df = df[~df.isnull().any(axis=1)]



# from scipy.interpolate import interp1d
# # original deta preprocess
# totoal_original_mask = 0 + df.isnull().values

# content = np.array(df)
# content[8758,13] = 60.0
# content[0, 29] = 78.0

# y_hat = []
# total_mask = totoal_original_mask
# total_mask[0] = 0
# total_mask[total_mask.shape[0] - 1] = 0
# mask_seq = [i for i in range(total_mask.shape[0])]
# mask_seq = np.array(mask_seq)
# for kk in range(total_mask.shape[1]):
#     x = []
#     y = []
#     for ii in range(total_mask.shape[0]):
#         if total_mask[ii, kk] == 0:
#             x.append(mask_seq[ii])
#             y.append(content[ii, kk])
#     f = interp1d(x, y)
#     y_hatt = f(mask_seq)
#     y_hat.append(y_hatt)
# data_seq1 = np.transpose(np.array(y_hat))

# df.loc[:] = data_seq1




# ############################################################################
# ############################################################################
# ############################################################################
# import numpy as np
# import pandas as pd
# import torch
# import scipy.sparse as sp

# from sklearn.metrics.pairwise import haversine_distances


# def geographical_distance(x=None, to_rad=True):
#     _AVG_EARTH_RADIUS_KM = 6371.0088

#     # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
#     latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

#     # If the input values are in degrees, convert them in radians
#     if to_rad:
#         latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

#     distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

#     # Cast response
#     if isinstance(x, pd.DataFrame):
#         res = pd.DataFrame(distances, x.index, x.index)
#     else:
#         res = distances

#     return res


# def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
#     if theta is None:
#         theta = np.std(x)
#     weights = np.exp(-np.square(x / theta))
#     if threshold is not None:
#         mask = x > threshold if threshold_on_input else weights < threshold
#         weights[mask] = 0.
#     return weights


# def get_similarity_AQI(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
#     theta = np.std(dist[:36, :36])  # use same theta for both air and air36
#     adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
#     if not include_self:
#         adj[np.diag_indices_from(adj)] = 0.
#     if force_symmetric:
#         adj = np.maximum.reduce([adj, adj.T])
#     if sparse:
#         import scipy.sparse as sps
#         adj = sps.coo_matrix(adj)
#     return adj


# def get_adj_AQI36():
#     df = pd.read_csv("./data/pm25/SampleData/pm25_latlng.txt")
#     df = df[['latitude', 'longitude']]
#     res = geographical_distance(df, to_rad=False).values
#     adj = get_similarity_AQI(res)
#     return adj

# def mask_without_nan(a,b):
#     mask = ~np.logical_or(np.isnan(a), np.isnan(b))
#     return mask

# def corr(a,b):
#     mask = mask_without_nan(a,b)
#     coorelation = np.corrcoef(a[mask],b[mask])
#     return coorelation[0,1]

# adj = get_adj_AQI36()
# # original deta preprocess
# totoal_original_mask = 0 + df.isnull().values

# content = np.array(df)

# content_ = np.zeros([8763,36])
# content_[2:-2] = np.array(df)
# content_[0] = content[0]
# content_[1] = content[0]
# content_[-1] = content[-1]
# content_[-2] = content[-1]


# total_mask = np.zeros([8763,36])
# total_mask[2:-2] = totoal_original_mask

# # window_space = np.zeros([36])
# window_time = np.array([.15,.35,0,.35,.15]) * .5
# for i in range(content.shape[1]):
#     window_space = adj[i]/adj[i].sum()*.5
#     for j in range(content.shape[0]):
#         if total_mask[j+2,i]==1:
#             mm1 = np.array(total_mask[j+2,:], dtype=bool)
#             window_space_ = window_space[~mm1]
#             w_space = 0
#             if window_space_.sum()>0:
#                 window_space_ = window_space_/window_space_.sum()*.5
#                 w_space = (content_[j+2,:][~mm1] * window_space_).sum()
            
#             mm2 = np.array(total_mask[j:j+5,i], dtype=bool)
#             window_time_  = window_time[~mm2]
#             w_time = 0
#             if window_time_.sum()>0:
#                 window_time_  = window_time_/window_time_.sum()*.5
#                 w_time = (content_[j:j+5,i][~mm2] * window_time_).sum()
            
#             content[j,i] = w_time + w_space
            
#             # y_hat
# data_seq1 = content

# df.loc[:] = data_seq1
# df[np.isnan(df)] = 0




df = np.array(df)
# df[np.isnan(df)] = 0





correlation = np.corrcoef(np.array(df).T)

import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram

corr_values = correlation
d = sch.distance.pdist(corr_values)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.6*d.max(), 'distance')

df_reindex = np.array([df[:,i] for i in np.argsort(ind)]).T

print(ind)
print(np.argsort(ind))
dendrogram(L)

correlation_reindex = np.corrcoef(df_reindex.T)

fig, ax = plt.subplots(1,2,figsize=[20,10])
ax[0].imshow(correlation)
ax[1].imshow(correlation_reindex)
ax[1].set_xticks(np.arange(36)) 
ax[1].set_xticklabels(np.argsort(ind))
# fig.savefig('co.png')


# ind[ind==4]=2
aa = np.zeros([36,36])
mask = np.zeros([36,36], dtype=bool)

mask[ind==1] = True
mask[:, ind!=1] = False
aa[mask] = correlation[mask]
mask[ind==2] = True
mask[:, ind!=2] = False
aa[mask] = correlation[mask]
mask[ind==3] = True
mask[:, ind!=3] = False
aa[mask] = correlation[mask]
mask[ind==4] = True
mask[:, ind!=4] = False
aa[mask] = correlation[mask]
np.save('adj_corr.npy', aa)

