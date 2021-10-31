# Author: Chethan Jujjavarapu
# Purpose: Subset and clean the feature tables for the cohort patients

import pandas as pd
import numpy as np
import sys
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append("Code/")
from DxCodeHandler.Converter import Converter
from DxCodeHandler.ICD9 import ICD9
from DxCodeHandler.ICD10 import ICD10

class Preprocessing:

    # Parameters: cohortsDf, dataframe, contains the positive and negative patients which contains the patientID (
    #   strings of patient IDs), cohortDay (day for 2nd lss/ldh diagnosisDepthLevel3 code), decompression (day for decompression
    #   sugery or not), and Group (indicate if positive or negtive group); patientID, string, patient id column for all
    #   tables.
    # Exception: None
    # Purpose: creates the cohortDf variable available to all functions in this class. Changes the name of the
    #   data column to "cohortDay" for easier table manipulation within the subset() functions. For processing ICD
    #   codes, we use DxCodeHandler functions.
    # Return: None.
    def __init__(self, cohortsDf, patientID="patientID", cohortDay="dxDay"):

        # change day column to a different name
        cohortsDf.rename(columns = {cohortDay : "cohortDay"}, inplace=True)

        # add df as class variable
        self.cohortsDf = cohortsDf

        # columns of interest
        self.patientID = patientID
        self.cohortDay = "cohortDay"

        # DxCodeHandler
        self.con = Converter()
        self.icd9 = ICD9()
        self.icd10 = ICD10()
    
    ############ SUBSET

    # Parameters: diagnosisDf, dataframe, contains the diagnosisDepthLevel3-related encounter information; diagDay, str,
    #   day for diagnosisDepthLevel3.
    # Exception: None
    # Purpose/Return: subset the diagnosisDf for the cohort patients and return a dataframe.
    def subsetDiagnosis(self, diagnosisDf, diagDay="dxDay"):

        # subset for cohort patients
        diagnosisPatientsDf = pd.merge(diagnosisDf, self.cohortsDf[[self.patientID, self.cohortDay]], on=self.patientID,
                                       how="inner")

        # get difference
        diagnosisPatientsDf["difference"] = diagnosisPatientsDf[diagDay] - diagnosisPatientsDf[self.cohortDay]

        # subset
        diagnosisPatientsDf = diagnosisPatientsDf.loc[diagnosisPatientsDf["difference"] <= 0]

        # remove columns
        del diagnosisPatientsDf[self.cohortDay]
        del diagnosisPatientsDf["difference"]

        return diagnosisPatientsDf

    # Parameters: procedureDf, dataframe, contains the procedure-related encounter information; procDay, str,
    #   day for procedure.
    # Exception: None
    # Purpose/Return: subset the procedureDf for the cohort patients and return a dataframe.
    def subsetProcedure(self, procedureDf, procDay="procDay"):

        procedurePatientsDf = pd.merge(procedureDf, self.cohortsDf[[self.patientID, self.cohortDay]], on=self.patientID,
                                       how="inner")

        procedurePatientsDf["difference"] = procedurePatientsDf[procDay] - procedurePatientsDf[self.cohortDay]

        procedurePatientsDf = procedurePatientsDf.loc[procedurePatientsDf["difference"] <= 0]

        del procedurePatientsDf[self.cohortDay]
        del procedurePatientsDf["difference"]

        return procedurePatientsDf

    # Parameters: demographicsDf, dataframe, contains the demographics.
    # Exception: None
    # Purpose/Return: subset the demographicsDf for the cohort patients and return a dataframe.
    def subsetDemographics(self, demographicsDf):

        return demographicsDf.loc[demographicsDf[self.patientID].isin(self.cohortsDf[self.patientID].unique())]

    # Parameters: prescriptionDf, dataframe, contains the prescription information
    # Exception: None
    # Purpose/Return: subset the prescriptionDf for the cohort patients and returns a dataframe.
    def subsetPrescription(self, prescriptionDf, prescriptionDay="prescribedDay"):

        prescriptionPatientsDf = pd.merge(prescriptionDf, self.cohortsDf[[self.patientID, self.cohortDay]], on=self.patientID,
                                       how="inner")

        prescriptionPatientsDf["difference"] = prescriptionPatientsDf[prescriptionDay] - prescriptionPatientsDf[self.cohortDay]

        prescriptionPatientsDf = prescriptionPatientsDf.loc[prescriptionPatientsDf["difference"] <= 0]

        del prescriptionPatientsDf[self.cohortDay]
        del prescriptionPatientsDf["difference"]

        return prescriptionPatientsDf

    # Parameters: reportsDf, dataframe, contains the index image reports.
    # Exception: None
    # Purpose/Return: subset the reportsDf for the cohort patients and return a dataframe.
    def subsetReports(self, reportsDf):

        return reportsDf.loc[reportsDf[self.patientID].isin(self.cohortsDf[self.patientID].unique())]

    # Parameters: df, dataframe; patientID, string; day, string; removalTechnique, string, indicates how to remove
    #   outliers. "Max" means you remove any patient in which count is great than Q3 + 1.5*(IQR) or "IQR" means you
    #   remove patients less than Q1 or greater than Q3 of the count distribution.
    # Exception: None
    # Purpose/Return: return a list of patients whose # of encounters is less than or equal to Max value (Q3 + 1.5*IQR)
    def encounterCountOutlierRemoval(self, df, patientID="patientID", day="dxDay", removalTechnique="Max"):

        frequencyDf = df.loc[:, [patientID, day]].drop_duplicates().loc[:, patientID].value_counts().reset_index()
        frequencyDf.columns = [patientID, "count"]
        frequency = df.loc[:, [patientID, day]].drop_duplicates().loc[:, patientID].value_counts().values.tolist()
        distribution = self.getDistributionSummary(frequency)
        if removalTechnique == "Max":
            patients = frequencyDf.loc[frequencyDf["count"] <= distribution["Max"], :].loc[:, patientID].tolist()
        elif removalTechnique == "Q1-Max":
            patients = frequencyDf.loc[(frequencyDf["count"] <= distribution["Max"]) & (frequencyDf["count"] >= distribution["Q1"]), :].loc[:, patientID].tolist()
        else:
            patients = frequencyDf.loc[(frequencyDf["count"] <= distribution["Q3"]) & (frequencyDf["count"] >= distribution["Q1"]), :].loc[:, patientID].tolist()
        return(patients)

    # Parameters: df, dataframe; patientID, string; day, string; binSize, int.
    # Exception: None
    # Purpose/Return: return a dataframe that has bins for each patient
    def addBins(self, df, patientID="patientID", day="dxDay", binSize=3):
        dfList = []
        patients = df.loc[:, patientID].unique()
        for patient in patients:
            dfPatient = df.loc[df.loc[:, patientID] == patient, ]
            uniqueEncounters = dfPatient.loc[:, day].unique()
            encBinMap = self.mapEncountersToBins(uniqueEncounters, binSize)
            dfPatient.loc[:, "Bin"] = dfPatient.loc[:, day].replace(encBinMap)
            dfList.append(dfPatient)
        dfWithBins = pd.concat(dfList)
        return(dfWithBins)

    # Parameters: diagnosisCohortsDf, dataframe, contains diagnosisDepthLevel3 information for the positive and negative patients;
    #             depthLevel, int, indicates the ICD level to map all valide codes to.
    # Exception: None
    # Purpose: maps all ICD-10 and ICD-9 codes to ICD-9 at a specific depth value for the diagnosisDepthLevel3 table.
    # Return: dataframe.
    def standardizeDiagnosis(self, diagnosisCohortsDf, depthLevel=2):
        
        # convert codes to string
        diagnosisCohortsDf["dx"] = diagnosisCohortsDf["dx"].apply(str)

        # map from icd-10 to icd-9 specific depth level
        diagnosisCohortsDf_icd10 = diagnosisCohortsDf.loc[diagnosisCohortsDf.dxTypeID == 10, :]
        ## get the mapped icd9 codes
        icd_10_9 = {}
        icd10Codes = diagnosisCohortsDf_icd10.dx.unique().tolist()
        for code in icd10Codes:
            try:
                # convert to icd-9
                icd9Codes = self.con.convert_10_9(code)
                # convert icd-9 to depthLevel
                icd9Codes = self.icd9.abstract(icd9Codes, depthLevel)
                icd9Codes = list(set(icd9Codes))
                icd_10_9[code] = icd9Codes
            except:
                # invalid ICD-10 code
                ## get depth 3 from ICD-10 code
                depth3Code = code.split(".")[0]
                icd_10_9[code] = depth3Code
        ## update the df
        list_of_dfs = []
        for i, row in diagnosisCohortsDf_icd10.iterrows():
            icd9Codes = icd_10_9.get(row["dx"])
            row = row.to_frame().T
            rows = pd.concat([row] * len(icd9Codes))
            rows["dx"] = icd9Codes
            list_of_dfs.append(rows)
        diagnosisCohortsDf_icd10_mapped = pd.concat(list_of_dfs)

        # map icd-9 to specific depth level
        diagnosisCohortsDf_icd9 = diagnosisCohortsDf.loc[diagnosisCohortsDf.dxTypeID == 9, :]
        icd_9_depth = {}
        icd9Codes = diagnosisCohortsDf_icd9.dx.unique().tolist()
        for code in icd9Codes:
            try:
                # convert icd-9 to depthLevel
                codes = self.icd9.abstract(code, depthLevel)
                codes = list(set(codes))
                icd_9_depth[code] = codes
            except:
                # invalid ICD-9 code
                ## get depth 3 from ICD-10 code
                depth3Code = code.split(".")[0]
                icd_9_depth[code] = depth3Code
        list_of_dfs = []
        ## update the df
        for i, row in diagnosisCohortsDf_icd9.iterrows():
            icd9Code = icd_9_depth.get(row["dx"])
            row = row.to_frame().T
            row["dx"] = icd9Code
            list_of_dfs.append(row)
        diagnosisCohortsDf_icd9_mapped = pd.concat(list_of_dfs)

        # combined mapped data
        diagnosisCohortsDf_mapped = pd.concat([diagnosisCohortsDf_icd10_mapped, diagnosisCohortsDf_icd9_mapped])
        del diagnosisCohortsDf_mapped["dxTypeID"]

        return diagnosisCohortsDf_mapped

    # def cleanPrescription():
    ## performed using Experiments/2021-01-13_Data_Shaping/Code/Development/2021-02-22/main_prescriptions.py

    # Parameters: procedureCohortDf, dataframe, procedures for patients in the cohort
    # Exception: None
    # Purpose: Subset for CPT, HCPCS Level II, ICD-9-PCS, ICD-10-PCS encounters
    # Return: a cleaned procedure dataframe
    def standardizeProcedure(self, procedureCohortDf, typeID="procTypeID"):
        return procedureCohortDf.loc[procedureCohortDf[typeID].isin([1,2,9,8,10]), :]

    # Parameters: indexReportsDf, dataframe, index reports for patients; textColumns, list, column names for text columns.
    # Exception: None
    # Purpose: Combine text coumns and clean text
    # Return: a cleaned index reports.
    def standardizeReports(self, indexReportsDf, textColumns=["findings", "impression"]):

        # combine text columns
        indexReportsDf[textColumns[0]] = indexReportsDf[textColumns[0]].replace(np.nan, " ", regex=True)
        indexReportsDf[textColumns[1]] = indexReportsDf[textColumns[1]].replace(np.nan, " ", regex=True)
        indexReportsDf["combinedText"] = indexReportsDf[textColumns[0]].astype(str) + " " + indexReportsDf[textColumns[1]].astype(str)

        # subset
        indexReportsSubsetDf = indexReportsDf.loc[:, [self.patientID, "combinedText"]]

        # clean
        indexReportsSubsetDf["text"] = indexReportsSubsetDf["combinedText"].apply(self.cleanText)

        # subset
        indexReportsSubsetDf = indexReportsSubsetDf.loc[:, [self.patientID, "text"]]

        return(indexReportsSubsetDf)    
    
    ############ HELPER FUNCTIONS

    # Parameters: arr, list of values
    # Exception: None
    # Purpose: Calculate the distribution of values in list
    # Return: Min, Q1, Median, Q3, Max of list of values
    def getDistributionSummary(self, arr):

        # sort list
        arr.sort(reverse=True) 
        # 25th percentile
        lowerP = np.nanpercentile(arr, 25)
        # 50th percentile
        middleP = np.nanpercentile(arr, 50)
        # 75th percentile
        higherP = np.nanpercentile(arr, 75)
        # IQR
        iqr = higherP - lowerP
        # min
        minP = lowerP - 1.5*iqr
        # max
        maxP = higherP + 1.5*iqr
        # mean
        mean = np.mean(arr)
        # sd
        sd = np.std(arr)
        # results
        results = {"Min" : minP,
                "Q1" : lowerP,
                "Median" : middleP,
                "Mean" : mean,
                "Q3" : higherP,
                "Standard_Dev" : sd,
                "Max" : maxP}
        return(results)

    # Paremters: text, string.
    # Exeption: None
    # Purpose: clean text using approach described in this website: www.analyticsvidhya.com/blog/2020/11/text-cleaning-nltk-library/
    # Return: a cleaned text, string.
    def cleanText(self, text):

        # check for nan
        if text == text:

            # remove extra white space and punctuation
            text = re.sub("[^A-Za-z0-9]+", " ", text)

            # text normalization
            ## control case
            text = text.lower()
            ## tokenize
            words = nltk.tokenize.word_tokenize(text)
            ## remove stopwords
            stopwords = nltk.corpus.stopwords.words("english")
            words = [i for i in words if i not in stopwords]
            ## stemming
            ps = nltk.PorterStemmer()
            words = [ps.stem(word) for word in words]

            # create cleaned text
            text = " ".join(words)
        
        else:

            text = "nan"

        return(text)

    # Parameters:
    #   df (dataframe): dataframe of interest to subset
    #   dayColumn (str): day column of the dataframe
    #   days (int): # of days to look behind the anchor day point
    # Exception: None
    # Purpose: Subsets dataframe based on time
    # Return: returns a subsetted dataframe
    def timeFiltration(self, df, dayColumn, days=90):

        upper = self.cohortsDf.loc[:, self.cohortDay].unique()[0]
        lower = upper - days
        dfSubset = df.loc[(df.loc[:, dayColumn] < upper) & (df.loc[:, dayColumn] >= lower), :]
        return(dfSubset)










    





















