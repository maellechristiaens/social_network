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

#create anonymous function to use them in other functions, to avoid 
#writting several if loops
func_centrality = lambda x : nx.eigenvector_centrality(x)

func_constraint = lambda x : nx.constraint(x)

func_social_distance = lambda x : dict(nx.all_pairs_shortest_path_length(x))

func_soc_dist_ind = lambda x,y,z : nx.shortest_path_length(x, y.index(z))


def matrices_infos(list_ind, infos, which):
    """ 
    MC 14/06/21
    To compute dissimilarity of infos (gender, domination, age) between 
    the individuals of a colony based on an excel sheet where are stored 
    these infos for each individual
    
    Inputs
        list_ind : list of the individuals of interest
        infos : pandas dataframe of the infos 
        which : which infos the matrix is from (gender, age or domination)
    Outputs
        dissimilarity matrix of the infos (gender, age or domination)
    """
    l = len(list_ind)
    matrix = np.zeros(shape=(l, l)) #create a square null matrix of size=nb of individuals considered
    for i in range(l): #for each column
        for j in range(l): #for each row
            if which == 'Age':
                matrix[i][j]=abs(infos.Age[i]-infos.Age[j]) #the value of the (i,j)th element of the matrix is the difference between the value of the ith subject and the jth subject
            if which == 'Domination':
                matrix[i][j]=abs(infos.Domination[i]-infos.Domination[j])
            if which == 'Gender':
                matrix[i][j]=abs(infos.Gender[i]-infos.Gender[j])
    return matrix
                
def binary_matrix(matrix):
    """
    MC 07/04/21
    To binarize a weighted matrix
    
    Inputs:
        matrix to binarize
    Outputs:
        binary matrix
    """
    s = len(matrix)
    mb = np.zeros(shape=(s, s)) #initialisation of the matrix, where all elements are 0 
    for i in range(s):
        for j in range(s):
            if matrix[i][j] != 0 : #if the (i,j)th element of the initial matrix is different from 0
                mb[i][j] = 1 #the (i,j)th element of the binarized matrix is equal to 1
    return mb


def thresholed_matrix(matrix,step):
    """
    MC 24/03/21
    To find a threshold such that the mean number of direct connexions 
    for each individual is 3. All the weaker connexions are discarded
    
    Inputs:
        matrix : matrix to threshold
        step : the step by which to increase the threshold
    Outputs:
        matrix thresholed, threshold
    """
    mat = matrix
    g = nx.Graph(mat)
    d = g.degree() #compute the degree of the matrix (ie the number of connexions each individual has)
    m = np.mean([v for k, v in d]) #compute the mean degree of the colony
    t = 0 #initialisation of the threshold
    while m > 3: #while the mean degree is above 3, increase the threshold by the step given by the user
        t += step
        for i in range(len(mat)):
            for j in range(len(mat)): #look at every dyadic interaction level
                if mat[i][j] < t : #if a dyadic interaction level is below the threshold
                    mat[i][j] = 0 #discard this interaction (setting it up to 0)
                    g = nx.Graph(mat)
                    d = g.degree() 
                    m = np.mean([v for k, v in d]) #recompute the mean degree of the new matrix
                    
    return mat, t #return the matrix thresholded and the threshold found 


def dis_matrix_global(matrix, individuals, function):
    """ 
    MC 23/03/21
    To compute the dissimilarity matrix for the metrics of interest 
    (Social distance, centrality or constraint)
    
    Inputs
        matrix : matrix from which to compute dissimilarity
        individuals : list of all the individuals in the colony
        function : a lambda function (func_social_distance, func_centrality 
        or func_constraint) 
    Outputs
        dissimilarity matrix (numpy array)
    """
    mat = np.zeros(shape=(len(individuals), len(individuals))) #initialisation of the dissimilarity matrix
    g = nx.Graph(matrix) #allow to compute metrics about the network
    all_values = function(g) #apply the function given by the user
    for i in range(len(mat)): #for all element of the matrix
        for j in range(len(mat)):
            if str(function) == str(func_social_distance):
                mat[i,j] = all_values[i][j] #compute the values, depending on the metric choosen
            else:
                mat[i,j] = abs(all_values[i]-all_values[j])
    return mat


def dis_matrix_individual(matrix, individuals, id_scan, function):
    """ 
    MC 23/03/21
    Inputs
        matrix : matrix from which to compute dissimilarity
        individuals : list of all the individuals in the colony
        id_scan : name of the individual scanned
        function : a lambda function (func_social_distance, func_centrality 
        or func_constraint) or 'Kinship'
    Outputs
        dissimilarity matrix (numpy array)
    """
    dis_matrix = np.zeros(shape=(len(individuals), len(individuals))) #initialisation of the dissimilarity matrix
    index = individuals.index(id_scan) #store the index in the list of individuals of the individual of interest
    if function == 'Kinship': 
        values = matrix[index]
    else :
        g = nx.Graph(matrix) #allow to compute metrics about the network
        if str(function) == str(func_soc_dist_ind):
            values = function(g, individuals, id_scan)
        else:
            all_values = function(g)
            value = all_values[index] #different ways to compute the metric depending on which one it is 
    for i in range(len(dis_matrix)): #for each element of the matrix 
        for j in range(len(dis_matrix)):
            if function=='Kinship' or str(function)==str(func_soc_dist_ind):
                dis_matrix[i,j] = abs(values[i]-values[j]) #take the absolute difference between the value of the individual of interest and the other individuals
            else:
                dis_matrix[i,j] = abs((all_values[i]-value)-(all_values[j]-value))
    dis_matrix = np.delete(dis_matrix, [index, index] 1) #delete the row and column of the individual of interest
    return dis_matrix
    
    



def dsi_aggressive(list_of_b, ind, fichiers, rand=False,giv=None, rece=None):
    """
    MC 07/04/21
    Inputs :
        list_of_b : list of behaviors on which to compute the DSI
        ind : individus from which to compute the DSI
        rand : True if we want to calculate random matrices for bootstrap, by default = False
        giv : givers individus from which to compute the DSI
        rece : receivers individus from which to compute the DSI
        fichiers : where the data are stored (by default = fichiers)
        
    Outputs:
        matrix of DSI for each dyad
    """
    if rand == False:
        giv = ind
        rece = ind
    means_b = np.zeros(shape=(1,len(list_of_b)))
    matrices_b = { str(i) : np.zeros(shape=(len(ind), len(ind))) for i in list_of_b}
    matrix = np.zeros(shape=(len(ind), len(ind)))
    total = { str(i) : 0 for i in list_of_b}
    for fichier in fichiers :
        data=pd.read_csv(fichier, sep=';', encoding="latin-1")
        for rang in range(len(data)):
            givers = []
            receivers = []
            if data.Behavior[rang] in list_of_b :
                if 'Focal est recepteur' in str(data.Modifiers[rang]):
                    for i in ind:
                        if i in str(data.Modifiers[rang]):
                            givers.append(i)
                    receivers.append(data.Subject[rang])
                if 'Focal est emetteur' in str(data.Modifiers[rang]):
                    givers.append(data.Subject[rang])
                    for i in ind:
                        if i in str(data.Modifiers[rang]):
                            receivers.append(i)
            for i in givers:
                for j in receivers:  
                    matrices_b[data.Behavior[rang]][giv.index(i),rece.index(j)]+=1
                    total[data.Behavior[rang]] += 1 
    for b in list_of_b:
        matrices_b[b] = (matrices_b[b]/total[b])        
        means_b[0][list_of_b.index(b)] = np.mean(matrices_b[b])
        matrices_b[b] = matrices_b[b]/(means_b[0][list_of_b.index(b)])
        matrix += matrices_b[b]
        
    matrix = matrix/len(list_of_b)
    return matrix


def dsi_affiliative(list_of_b, ind, files):
    """
    MC 07/04/21 
    Inputs :
        list_of_b : list of behaviors on which to compute the DSI
        ind : individus from which to compute the DSI
        files : where the data are stored
        
    Outputs:
        matrix of DSI for each dyad
    """
    matrices_b = { str(i) : np.zeros(shape=(len(ind), len(ind))) for i in list_of_b} #initialisation of all the interaction matrices
    matrices_nb_oc = { str(i) : np.zeros(shape=(len(ind), len(ind))) for i in list_of_b} #initialisation of the number of occurence of each behavior
    matrix = np.zeros(shape=(len(ind), len(ind))) #initialisation of the final DSI matrix 
    total = { str(i) : 0 for i in list_of_b} #initialisation of the total duration of each behavior
    nb_events = { str(i) : 0 for i in list_of_b} #initialisation of the total number of each behavior
    for file in files : #for each file where the data are stored
        data=pd.read_csv(file, sep=';', encoding="latin-1") #read the file
        for rank in range(len(data)): #for each line of data
            focal = data.Subject[rank] #store which individual is in focal
            if data.Behavior[rank] == '1 Debut Grooming' and data.Modifiers[rank]!='None': #if the behavior of this line is a start of grooming (and not an error)
                start = data['Start (s)'][rank] #store the time of the beginning of this grooming
                end=0 #reset the time of the end of this grooming
                rank2 = rank #initialisation of the line number from which to look for the end of the grooming
                for i in range(rank, len(data)): #for all the lines after the line considered
                    if data.Behavior[i]=='2 Zone de Grooming': 
                        groomed = re.findall(r'\d+', data.Modifiers[i]) #store the name of the individual participating in the grooming with the focal
                        rank2 = i #update the line number from which to look for the end of the grooming 
                        break #stop the for loop once the grooming partner has been found
                for j in range (rank2, len(data)): #for all the lines after the line considered
                    if data.Behavior[j] == '4 Fin Grooming' and data.Subject[j] == focal: #if it's an end of grooming and we're still with the same focal
                        who = re.findall(r'\d+', data.Modifiers[j]) #look who is the subject of this grooming
                        if who == groomed: #if it's the same as the one stored before (because several individuals can participate in the same episod of grooming)
                            end = data['Stop (s)'][j] #update the time of the end of this grooming
                            break #stop the for loop once the end time has been found
                if end !=0 and end>start: #if an end has been found and this end is after the beginning
                    duration = end - start #store the duration of the grooming episod
                else:
                    duration = 0 #otherwise set the duration to 0

                for i in ind: #for all the individuals in the colony
                    if i in str(data.Modifiers[rank]): #if this individual is the subject of this grooming
                        matrices_b['1 Debut Grooming'][ind.index(i),ind.index(focal)]+=duration 
                        matrices_nb_oc['1 Debut Grooming'][ind.index(i),ind.index(focal)]+= 1
                        matrices_b['1 Debut Grooming'][ind.index(focal),ind.index(i)]+=duration
                        matrices_nb_oc['1 Debut Grooming'][ind.index(focal),ind.index(i)]+= 1
                        total['1 Debut Grooming'] += duration
                        nb_events['1 Debut Grooming'] += 1


            elif data.Behavior[rank] in list_of_b and data.Behavior[rank] != '1 Debut Grooming':
                for i in ind:
                    if i in str(data.Modifiers[rank]):
                        matrices_b[data.Behavior[rank]][ind.index(i),ind.index(focal)]+=data['Duration (s)'][rank]
                        matrices_nb_oc[data.Behavior[rank]][ind.index(i),ind.index(focal)]+= 1
                        matrices_b[data.Behavior[rank]][ind.index(focal),ind.index(i)]+=data['Duration (s)'][rank]
                        matrices_nb_oc[data.Behavior[rank]][ind.index(focal),ind.index(i)]+= 1
                        total[data.Behavior[rank]] += data['Duration (s)'][rank]
                        nb_events[data.Behavior[rank]] += 1
                        
    for b in list_of_b:
        matrices_b[b][matrices_b[b] != 0] = matrices_b[b][matrices_b[b] != 0]/matrices_nb_oc[b][matrices_nb_oc[b] != 0]
        mean = total[b]/nb_events[b]
        matrices_b[b] = matrices_b[b]/(mean)
        matrix += matrices_b[b]
        
    matrix = matrix/len(list_of_b)
    return matrix


def matrix_grooming(ind, fichiers, symetrical=False):
    """
    MC 02/06/21 
    Inputs :
        ind : individus from which to compute the grooming
        fichiers : where the data are stored 
        symetrical : whether we want a symetrical (True) or a directed (False) matrix
        
    Outputs:
        matrix of grooming for each dyad
    """
    
    matrix = np.zeros(shape=(len(ind), len(ind))) #Initiation
    for fichier in fichiers :
        print(fichier)
        data=pd.read_excel(fichier) #Regarde les focaux un par un

        for rang in range(len(data)): #Regarde le focal ligne par ligne 
            focal = data.Subject[rang] #Pour chaque ligne, regarde l'individu pris en focal
            if data.Behavior[rang] == '1 Debut Grooming' and data.Modifiers[rang]!='None': #si le behavior est du début de grooming
                start = data['Start (s)'][rang] #Stocker le temps de départ de ce grooming
                end=0
                for i in range(rang, len(data)):
                    if data.Behavior[i]=='2 Zone de Grooming':
                        groomed = re.findall(r'\d+', data.Modifiers[i])
                        rang2 = i
                        break
                for j in range (rang2, len(data)):
                    if data.Behavior[j] == '4 Fin Grooming':
                        who = re.findall(r'\d+', data.Modifiers[j])
                        if who == groomed:
                            end = data['Stop (s)'][j]
                            break
                if end !=0:
                    duration = end - start
                else:
                    duration = 0 

                for i in ind:
                    if i in str(data.Modifiers[rang]):
                        matrix[ind.index(i),ind.index(focal)]+=duration
                        if symetrical:
                            matrix[ind.index(focal),ind.index(i)]+=duration
    return matrix

