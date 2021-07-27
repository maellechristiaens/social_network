import pandas as pd
import numpy as np
import os

def go_through(matrix):
    """
    MC 28/06/2021
    This function creates an iterator to go through each element of the 
    matrix (helpful to reduce the number of rows in the code)
    Inputs:
        matrix: the matrix we want to go through
    Outputs:
        An iterable of the matrix
    
    """
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            yield matrix[i][j]


def list_files(path):
    """
    MC 14/06/21
    Inputs:
        path : path of the folder were the focals are stored 
    Outputs:
        list of all the paths to all the files
    """
    fichiers = []
    for (_,_,files) in os.walk(path): #look at all the files in the folder given by the path
        for file in files:
            if file.endswith(".xls"):
                fichiers.append(os.path.join(path,file)) #add the path of each file in a list 
    return fichiers

def infos(path):
    """
    MC 14/06/21
    Inputs:
        path : path of the excel file were the infos are stored
    Outputs:
        table of the infos about the individuals 
    """
    infos = pd.read_excel(path, index_col='Individuals', sheet_name = 'Infos') #read the sheet named Infos in the excel given by the path 
    return infos

def kinship(path):
    """
    MC 15/06/21
    Inputs:
        path : path of the excel file were the infos about kinship are stored
    Outputs:
        matrix of kinship 
    """
    matrix = pd.read_excel(path, sheet_name='Kinship', index_col=0) #read the kinship sheet of the excel given by the path
    #matrix = pd.DataFrame.to_numpy(matrix) #transform the pd DataFrame to a numpy matrix
    return matrix

def reorder_files(fichiers, save = False, name=None, path=None):
    """
    MC/BK 29/04/21
    Inputs:
        fichiers : list of the paths to the files to reorder
        save : if you want to save the reorder into a new .csv, by default = False
        name : name of the new .csv, by default = None 
        path : where to save the file, by default = None
    Outputs:
        DataFrame of the files reordered, and a new .csv if save=True 
    """
    data = pd.read_excel(fichiers[0], encoding="latin-1") #read the first file of the list to get the column names
    data_complete = pd.DataFrame(columns=data.columns) #create an empty dataframe with only the columns, where all reordered files will be concatenated
    data_tmp = pd.DataFrame(columns=data.columns) #initialisation of the temporary dataframe
    for fichier in fichiers: #for each file in the list of files
        data = pd.read_excel(fichier, encoding="latin-1") #Read the file
        data_reordered = pd.DataFrame(columns=data.columns) #create a new empty dataframe with the same columns as the file to reorder
        for i in np.unique(data['Observation date']): #for each different date of observation (if there are several)
            data_day = data[data['Observation date']==i] #takes only the data from this date of observation
            for j in np.unique(data_day['Subject']): #for each subject followed during this day 
                data_day_subject = data_day[data_day['Subject']==j] #takes only the data from this subject followed
                order = np.argsort(data_day_subject['Start (s)']) #sort the rows of this data depending on their starting times
                data_tmp = pd.concat([data_tmp,data_day_subject.iloc[order]]) #concatenate the data following the order of the starting times
        data_complete = pd.concat([data_complete, data_tmp]) #concatenate the reordered files in one big dataframe
    if save:
        data_complete.to_csv(path + name, sep=';', encoding="latin-1")
    return data_complete