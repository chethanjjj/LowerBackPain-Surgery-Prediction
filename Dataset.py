import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import numpy as np

class DecompressionDataset(Dataset):

    # Parameters:
    #   textInput (dataframe): text data, rows - patients, column - list of indexed values from word2vec model
    #   timeInput (dataframe): temporal data, rows - patients and bin-level, columns - diagnosis, procedures, and prescriptions (on-hot encoded features)
    #   staticInput (dataframe): demographics data, rows - patients, columns - one-hot encoded demographic features
    #   cohorts (dataframe): patients grouped into positive and negative
    #   patientIDs (list): patientIDs for train/test set
    #   patientName (str): name of patientID column
    #   timeName (str): name of time (i.e. bin) column
    #   labelName (str): name of the group (positive or negative) column
    # Exception: None
    # Purpose: function assumes patients across dataframes are the same. Stores data into a PyTorch Dataset for easy access in deep learning model
    # Return: None
    def __init__(self, 
                 textInput, 
                 timeInput, 
                 staticInput, 
                 cohorts,
                 patientIDs,
                 patientName="patientID", 
                 timeName="Bin", 
                 textName="Indexed_Text",
                 labelName="Group_Encoded"):

        # subset for train/test patients
        cohorts = cohorts.loc[cohorts[patientName].isin(patientIDs)]
        #print("# of Patients Before Intersection: " + str(len(cohorts.patientID.unique())))

        # get common patients
        temporalPatientSet = set(timeInput[patientName].values)
        #print("# of Temporal Patients Before Intersection: " + str(len(temporalPatientSet)))

        demographicsPatientsSet = set(staticInput[patientName].values)
        #print("# of Static Patients Before Intersection: " + str(len(demographicsPatientsSet)))

        indexReportsPatientsSet = set(textInput[patientName].values)
        #print("# of Index-Report Patients Before Intersection: " + str(len(indexReportsPatientsSet)))

        cohortsPatientSet = set(cohorts[patientName].values)
        patientIDSet = set.intersection(temporalPatientSet,
                                        demographicsPatientsSet,
                                        indexReportsPatientsSet,
                                        cohortsPatientSet)
        cohorts = cohorts.loc[cohorts[patientName].isin(list(patientIDSet))]
        #print("# of Patients After Intersection: " + str(len(cohorts.patientID.unique())))

        
        # subset for patients and sort
        cohorts = cohorts.sort_values(by=patientName)
        self.patients = cohorts.loc[:, patientName].values
        textInputSubset = textInput.loc[textInput.loc[:, patientName].isin(self.patients), :].sort_values(by=patientName)
        timeInputSubset = timeInput.loc[timeInput.loc[:, patientName].isin(self.patients), :].sort_values(by=[patientName, timeName])
        staticInputSubset = staticInput.loc[staticInput.loc[:, patientName].isin(self.patients), :].sort_values(by=patientName)
        
        # convert to tensors
        self.text = torch.from_numpy(np.vstack(textInputSubset.loc[:, textName].values)).long()
        totalEncounters = len(set(timeInputSubset[timeName]))
        time = torch.from_numpy(timeInputSubset.drop([patientName, timeName], axis=1).values).float()
        # (# of patients, # of bins, # of features)
        self.time = torch.reshape(time, (int(time.shape[0]/totalEncounters), totalEncounters, time.shape[1]))
        self.static = torch.from_numpy(staticInputSubset.drop(patientName, axis=1).values).float()
        self.labels = torch.from_numpy(cohorts.loc[:, labelName].values).long()

    # Parameters:
    #   index (int): index of interest
    # Exception: None
    # Purpose: identifies a specific index for all data
    # Return: tuple of tensors for specific index
    def __getitem__(self, index):
        return(self.text[index, :], 
               self.time[index, :, :],
               self.static[index, :],
               self.labels[index])

    # Parameters: None
    # Exception: None
    # Purpose/Return: returns total # of patients
    def __len__(self):
        return(len(self.labels))

    # Parameters: None
    # Exception: None
    # Purpose/Return: return the labels as a numpy array
    def __getLabels__(self):
        return(self.labels.numpy())
    
    # Parameters: None
    # Exception: None
    # Purpose/Return: return the labels as a numpy array
    def __getPatientIDs__(self):
        return(self.patients)

    # Parameters: None
    # Exception: None
    # Purpose/Return: returns total # of features
    def __featureDim__(self):
        return(self.static.shape[1])

