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
    
    fig, ax = plt.subplots(figsize=(7,7)) #create a new figure 
    if id_scan != None: #if you plot a dissimilarity matrix
        individuals = np.delete(individuals, individuals.index(id_scan)) #remove the ind scanned from the list 
    if labels==True: #if you want the names of the individuals to be plotted 
        labels = individuals #store their names 
    sns.heatmap(matrix,vmin=np.min(matrix), vmax=np.max(matrix),xticklabels=labels, yticklabels=labels, cmap=color, annot=an, linewidths=0.1, linecolor='black',  square=True, cbar_kws={'orientation': 'horizontal','shrink': 0.7}) #plot the heatmap, with all the parameters choosen
    ax.invert_yaxis() #invert the axis to get the diagonal from bottom left to top right instead of top left to bottom right
    if save == True: #if you want to save the figure created 
        plt.savefig(path + name_save + '.png', bbox_inches='tight', pad_inches = 0.5)
            

def network(matrix, individuals, node_colors, network_type, other_matrix=None, title='Social_Network', ind=False, images=False, path_images = None, save = False, path=None):
    """
    MC 24/03/21
    Inputs :
        matrix : the matrix on which the network is based
        individuals : list of all the individuals of the colony 
        node_colors : list of the colors of all the nodes 
        network_type : 'Constraint', 'Centrality', 'Social distance'
        other_matrix : if we want a layout based on another matrix than the affiliative matrix, by default = None
        title : title of the network (NO SPACE PLS), by default = 'Social_Network'
        ind : the individual on which to calculate the social distance, by default = False
        images : if you want the pictures of the individuals plotted on the graph, by default = False
        path_images : the path where you can find the pictures of the individuals if you want to plot them, by default = None
        save : True if you want to save the figure, by default = False
        path : path to the directory where to save the figure, by default = None 
        (if None, save in the same directory as the code)
    Outputs :
        plot of the network
        plot saved if save = True
    """
    if images: #if you want to plot the pictures of the individuals on the network
        files = [f for f in glob.glob(path_images + "*.png")] #store the pictures' paths
        img = [] #initialisation of the different pictures 
        for f in files:
            img.append(mpimg.imread(f)) #store the pictures in one variable
    if other_matrix == None: #if you want to plot the links and positions from the same matrix or not
        other_matrix=matrix
    dict_names = { i : individuals[i] for i in range(0, len(individuals))} #store the individuals' names in a dictonary
    fig, ax = plt.subplots(figsize=(20,20)) #create a new figure
    g = nx.Graph(other_matrix) #store the informations of the network for one of the matrices
    h = nx.Graph(matrix) #store the informations of the network for the other matrix. Note that h and g may be equal if only 1 matrix is given
    if network_type == 'Constraint': #compute the values and the color of the graph depending on the type specified in the parameters
        value = nx.constraint(h)
        color = 'Greens'
    if network_type == 'Centrality':
        value = nx.eigenvector_centrality(h)
        color = 'Oranges'
    if network_type == 'Social distance':  
        value = nx.single_source_shortest_path_length(h, individuals.index(ind))
        m = max(value.values())
        for i in value.values():
            i = m - i #reverse the values because it is computed in the opposite order 
        color = 'Purples'
    widths = h.edges()
    weights = [(h[u][v]['weight'])*0.5 for u,v in widths] #rescale the widths of the edges
    pos=nx.kamada_kawai_layout(g) #choose a layout 
    cNorm  = colors.Normalize(vmin=min(value.values()), vmax=max(value.values()))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color) #normalize the colors to get the full spectrum 
    
    nx.draw_networkx_edges(h,pos) #draw the links between the individuals
    if images: #if you want to plot the pictures of the individuals 
        plt.axis('off') #remove the axis
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        ax=plt.gca()
        fig=plt.gcf() #get the informations about the axis and the figure to plot at the right positions 
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform #used to navigate between the different coordinate systems
        imsize = 0.05 #the size of the pictures plotted
        for n in g.nodes(): #for each individual
            (x,y) = pos[n] #get its position on the network graph
            xx,yy = trans((x,y))
            xa,ya = trans2((xx,yy)) #transform this position into coordinates of the figure
            b = plt.axes([xa-(imsize+0.02)/2.0,ya-(imsize+0.02)/2.0, imsize+0.02, imsize+0.02]) #define a local figure inside the big figure to display the colored rectangle
            z= np.zeros((img[n].shape[0]+50, img[n].shape[1]+50,3)) #create a rectangle around where the picture will be plotted
            colorVal = scalarMap.to_rgba(value[n]) #get the color for this picture
            z[:,:,0] = colorVal[0] 
            z[:,:,1] = colorVal[1]
            z[:,:,2] = colorVal[2] #redefine the color of the rectangle created to be the color associated with this picture
            b.imshow(z) #display the rectangle on the figure
            b.axis('off') #remove the axis from this rectangle
            a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize]) #define a local figure inside the big figure to display the picture
            a.imshow(img[n]) #display the picture on top of it
            a.set_aspect('equal')
            a.axis('off') #remove the axis from this picture
    else: #if you don't want to display the pictures on the network graph
        nx.draw_networkx_nodes(g, pos, node_color=node_colors) #only draw classical nodes of the color choosen
        label_options = {"fc": "white", "alpha": 0.8}
        nx.draw_networkx_labels(g, pos, labels=dict_names, bbox=label_options) #label these nodes with the names of the individuals 
        
    ax.margins(0.1, 0.1) #set some margins to see the whole network properly
    ax.set_title(title) #write a title
    if save == True: #if you want to save the figure
        if path == None: #if no special path has been given
            path = os.getcwd() #get the current path
        plt.savefig(path + title + '.png') #save the figure on the folder indicated by the path