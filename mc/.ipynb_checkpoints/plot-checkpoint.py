import numpy as np
import random
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import networkx as nx
import scipy
import os
import glob
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.cm as cmx
import re

from tqdm import tqdm
from mpl_toolkits import mplot3d
from scipy.stats import pearsonr
from scipy.stats import linregress


def heatmap(matrix, individuals, color, id_scan=None, save = False, name_save = None, path = 'C:/Users/maell/Documents/ENS/Cours/Césure/Stage_Sliwa/Strasbourg/Figures/', labels=True, an=False):
    """ 
    MC 24/03/21
    Inputs 
        matrix : matrix to be plotted
        individuals : list of all the individuals of the colony
        color : color of the map
        id_scan : name of the individual scanned if dissimilarity matrix, by default = None
        save : True if you want to save the picture, by default = False
        name_save : the name of the figure saved (be careful to give a name if you write save = True !)
        labels : if you want the name of the individuals to be plotted, by default = True
    Outputs
        plot of the heatmap
        plot saved if save = True 
    """
    
    fig, ax = plt.subplots(figsize=(7,7))
    if id_scan != None:
        individuals = np.delete(individuals, individuals.index(id_scan))
    if labels==True:
        labels = individuals
    sns.heatmap(matrix,vmin=np.min(matrix), vmax=np.max(matrix),xticklabels=labels, yticklabels=labels, cmap=color, annot=an, linewidths=0.1, linecolor='black',  square=True, cbar_kws={'orientation': 'horizontal','shrink': 0.7})
    ax.invert_yaxis()
    if save == True:
        plt.savefig(path + name_save + '.png', bbox_inches='tight', pad_inches = 0.5)
            
    
    
    

def network(matrix, individuals, node_colors, network_type, other_matrix=None, title='Social_Network', ind=False, images=False, save = False, path=None):
    """
    MC 24/03/21
    Inputs :
        matrix : the matrix on which the network is based
        individuals : list of all the individuals of the colony 
        node_colors : list of the colors of all the nodes 
        network_type : 'Constraint', 'Centrality', 'Social distance'
        other_matrix : if we want a layout based on another matrix than the affiliative matrix, by default = False
        title : title of the network (NO SPACE PLS), by default = 'Social_Network'
        ind : the individual on which to calculate the social distance, by default = False
        images : if you want the pictures of the individuals plotted on the graph, by default = False
        save : True if you want to save the figure, by default = False
        path : path to the directory where to save the figure, by default = None 
        (if None, save in the same directory as the code)
    Outputs :
        plot of the network
        plot saved if save = True
    """
    if images:
        path_images = 'C:/Users/maell/Documents/ENS/Cours/Césure/Stage_Sliwa/Strasbourg/Code/Rhesus/Individus/'
        files = [f for f in glob.glob(path_images + "*.png")]
        img = []
        for f in files:
            img.append(mpimg.imread(f))
    #if other_matrix == None:
        #other_matrix=matrix
    dict_names = { i : individuals[i] for i in range(0, len(individuals))}
    fig, ax = plt.subplots(figsize=(20,20))
    g = nx.Graph(other_matrix)
    h = nx.Graph(matrix)
    if network_type == 'Constraint':
        value = nx.constraint(h)
        color = 'Greens'
    if network_type == 'Centrality':
        value = nx.eigenvector_centrality(h)
        color = 'Oranges'
    if network_type == 'Social distance':  
        value = nx.single_source_shortest_path_length(h, individuals.index(ind))
        m = max(value.values())
        for i in value.values():
            i = m - i
        color = 'Purples'
    widths = h.edges()
    weights = [(h[u][v]['weight'])*0.5 for u,v in widths]
    pos=nx.kamada_kawai_layout(g)
    cNorm  = colors.Normalize(vmin=min(value.values()), vmax=max(value.values()))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color)
    
    nx.draw_networkx_edges(h,pos)
    if images:
        plt.axis('off')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        ax=plt.gca()
        fig=plt.gcf()
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        imsize = 0.05
        for n in g.nodes():
            (x,y) = pos[n]
            xx,yy = trans((x,y))
            xa,ya = trans2((xx,yy))
            b = plt.axes([xa-(imsize+0.02)/2.0,ya-(imsize+0.02)/2.0, imsize+0.02, imsize+0.02 ])
            z= np.zeros((img[n].shape[0]+50, img[n].shape[1]+50,3))
            colorVal = scalarMap.to_rgba(value[n])
            z[:,:,0] = colorVal[0]
            z[:,:,1] = colorVal[1]
            z[:,:,2] = colorVal[2]
            b.imshow(z)
            b.axis('off')
            a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
            a.imshow(img[n])
            a.set_aspect('equal')
            a.axis('off')
    else:
        nx.draw_networkx_nodes(g, pos, node_color=node_colors)
        label_options = {"fc": "white", "alpha": 0.8}
        nx.draw_networkx_labels(g, pos, labels=dict_names, bbox=label_options)
        
    ax.margins(0.1, 0.1)
    ax.set_title(title)
    if save == True:
        if path == None:
            path = os.getcwd()
        plt.savefig(path + title + '.png')