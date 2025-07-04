#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 13:50:44 2025

@author: dliu
"""

import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv("./data/pm25/SampleData/pm25_latlng.txt")
df = df[['latitude', 'longitude']]





color_list = np.arange(36)
color_list[[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = 0
color_list[[0,  1, 22, 23, 24, 26, 27, 28, 29, 31]] = 1
color_list[[25, 30, 32, 33, 34, 35]] = 2


df['color'] = color_list
df['size'] = 1

# Plot using scatter_map
fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    # hover_name="Address",
    # hover_data=["Address", "Listed"],
    color="color",
    color_continuous_scale="Viridis",
    # color_discrete_map={0: 'orange', 1: 'red', 2: 'blue'},
    size="size",
    size_max=15,
    zoom=8.6,
    height=800,
    width=800
)

# Use satellite basemap from MapLibre
# fig.update_traces(marker=dict(size=15))
fig.update_layout(mapbox_style="satellite")  # now works without a token
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_coloraxes(showscale=False)
fig.show()



# # Save as HTML
# fig.write_html("scatter_map_pms25.html")

# Save as PNG (requires kaleido)
fig.write_image("scatter_map_pms25.png")









df = pd.read_csv("./data/metr_la/sensor_locations_la.csv")
df = df[['latitude', 'longitude']]


color_list = np.arange(207)
color_list[[206,  91, 174,  87,  38, 170,  41, 165, 139, 163,  47,  48,  82,
        81, 160,  52, 159,  56, 141, 155,  60,  76,  63, 150,  73, 148,
       133,  95, 144, 191,  12, 127,  96,  15, 120, 193, 187, 100, 125,
         4,  25, 204,  23]] = 0
color_list[[75, 113, 111, 142,  74, 124, 140,  80, 119,
        115, 112,  94, 104, 205, 130, 138, 137, 101,  86, 131, 136,  72,
        134,  97,  93, 128,  89, 157,  69, 145,  26,  24, 186, 188,  19,
        190,  17, 181,  16, 195, 196, 197, 198,   6,   5,   3, 199, 192,
        180, 182,  33, 146, 149, 154,  61, 116,  57, 158,  54, 162, 161,
         43,  42, 166, 172, 175,  34, 164, 117]] = 1
color_list[[176, 122, 132, 185, 151,
        183, 103,  64,  84,   9,  88,   8,  14,  78,  98,  77, 102,   2,
         28,  59,  40,  70,  29,  68,  83,  79]] = 2
color_list[[189, 153, 152, 114,  50,
         45,  44, 167,  31, 179, 178,  21, 156,  35, 105,  10,  90, 121,
         203,  99,  18, 106, 126, 200]] = 3
color_list[[177,   1,  30, 202, 201,  13,  27,
          22,   7,  20, 194,  32, 184,  11,  49, 173, 110, 109, 123, 108,
         107, 129,  92, 135,  85, 143,  71, 147,  67,  66,  65,  62,  58,
          55,  53,  51, 118,  46, 168, 169,  39, 171,  37,  36,   0]] = 4


df['color'] = color_list
df['size'] = 1


color_scale = [(0, 'orange'), (1,'red'), (2,'blue')]
# Plot using scatter_map
fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    # hover_name="Address",
    # hover_data=["Address", "Listed"],
    color="color",
    color_continuous_scale="Viridis",
    # color_discrete_map=color_scale,
    size="size",
    size_max=10,
    zoom=10.5,
    height=800,
    width=800
)

# Use satellite basemap from MapLibre
fig.update_layout(mapbox_style="satellite")  # now works without a token
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_coloraxes(showscale=False)
fig.show()



# # Save as HTML
# fig.write_html("scatter_map_pms25.html")

# Save as PNG (requires kaleido)
fig.write_image("scatter_map_metrla.png")







df = pd.read_csv("../../data/pems_bay/sensor_locations_bay.csv", names=['id','latitude','longitude'])
df = df[['latitude', 'longitude']]


color_list = np.arange(325)
color_list[[0,   4,   5,  14,  19,  21,  23,  25,  27,  28,  29,  30,  34,
        35,  36,  40,  44,  45,  47,  52,  54,  57,  58,  59,  62,  64,
        66,  67,  69,  70,  71,  77,  78,  86,  87,  88,  89,  90,  92,
        93,  94,  98,  99, 100, 104, 105, 107, 110, 112, 118, 124, 125,
       128, 129, 130, 131, 132, 137, 138, 139, 142, 143, 149, 150, 156,
       157, 158, 159, 160, 161, 162, 165, 166, 168, 171, 172, 174, 181,
       184, 186, 187, 188, 189, 190, 193, 194, 198, 202, 204, 211, 213,
       214, 216, 218, 220, 222, 224, 228, 230, 237, 238, 239, 241, 243,
       246, 247, 250, 253, 255, 257, 258, 259, 260, 261, 262, 263, 264,
       268, 269, 270, 271, 272, 273, 280, 286, 287, 288, 289, 290, 291,
       292, 293, 294, 295, 296, 297, 298, 310, 312, 314, 315, 316, 320,
       321, 322, 323, 324]] = 0
color_list[[2,   6,  13,  16,  20,  38,  39,  43,  53,  63,  68,  80, 111,
       117, 140, 146, 148, 153, 154, 164, 170, 177, 180, 183, 185, 199,
       206, 207, 209, 210, 212, 215, 242, 251, 265, 266, 267, 313, 319]] = 1
color_list[[1,   3,   7,   8,   9,  10,  11,  12,  15,  17,  18,  22,  24,
        26,  31,  32,  33,  37,  41,  42,  46,  48,  49,  50,  51,  55,
        56,  60,  61,  65,  72,  73,  74,  75,  76,  79,  81,  82,  83,
        84,  85,  91,  95,  96,  97, 101, 102, 103, 106, 108, 109, 113,
       114, 115, 116, 119, 120, 121, 122, 123, 126, 127, 133, 134, 135,
       136, 141, 144, 145, 147, 151, 152, 155, 163, 167, 169, 173, 175,
       176, 178, 179, 182, 191, 192, 195, 196, 197, 200, 201, 203, 205,
       208, 217, 219, 221, 223, 225, 226, 227, 229, 231, 232, 233, 234,
       235, 236, 240, 244, 245, 248, 249, 252, 254, 256, 274, 275, 276,
       277, 278, 279, 281, 282, 283, 284, 285, 299, 300, 301, 302, 303,
       304, 305, 306, 307, 308, 309, 311, 317, 318]] = 2



df['color'] = color_list
df['size'] = 1


color_scale = [(0, 'orange'), (1,'red'), (2,'blue')]
# Plot using scatter_map
fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    # hover_name="Address",
    # hover_data=["Address", "Listed"],
    color="color",
    color_continuous_scale="Viridis",
    # color_discrete_map=color_scale,
    size="size",
    size_max=10,
    zoom=10.5,
    height=800,
    width=800
)

# Use satellite basemap from MapLibre
fig.update_layout(mapbox_style="satellite")  # now works without a token
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_coloraxes(showscale=False)
fig.show()



# # Save as HTML
# fig.write_html("scatter_map_pms25.html")

# Save as PNG (requires kaleido)
fig.write_image("scatter_map_pemsbay.png")