import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import nltk

class Featurize:

    def __init__(self, cohortsDf, patientID="patientID", cohortDay="cohortDay"):
        
        # change day column to a different name
        cohortsDf.rename(columns = {cohortDay : "cohortDay"}, inplace=True)

        # add df as class variable
        self.cohortsDf = cohortsDf

        # columns of interest
        self.patientID = patientID
        self.cohortDay = "cohortDay"

    # Parameters: demographicsCohortDf, dataframe, demographics for patients in the cohort; providerDf, dataframe,
    #   provider information
    # Exception: None
    # Purpose: one-hot encode categorical columns and normalize continuous columns
    # Return: a featurized demographics.
    def featurizeDemographics(self, demographicsCohortsDf, categoricalColumns=["providerGender", "providerType", "providerSpecialty", "imageTypeID", "genderID", "raceID", "ethnicityID"], continuousColumns = ["age"]):

        # subset
        cols = categoricalColumns + continuousColumns
        cols.append(self.patientID)
        demographicsCohortsDf = demographicsCohortsDf.loc[:, cols]
        
        ## one-hot encode
        for categoryColumn in categoricalColumns:
            one_hot = pd.get_dummies(demographicsCohortsDf[categoryColumn])
            one_hot = one_hot.add_prefix(categoryColumn + "_")
            #### this might change if i do a 95%-5% filtration. As a result, the dropped column may not be
            #### represented.
            one_hot = one_hot.drop(one_hot.columns[-1], axis=1)
            demographicsCohortsDf = demographicsCohortsDf.drop(categoryColumn, axis=1)
            demographicsCohortsDf = demographicsCohortsDf.join(one_hot)

        ## normalize continuous
        for continuousColumn in continuousColumns:
            mean = np.nanmean(demographicsCohortsDf[continuousColumn].values)
            demographicsCohortsDf[continuousColumn].fillna(value=mean, inplace=True)
            std = np.std(demographicsCohortsDf[continuousColumn].values)
            demographicsCohortsDf[continuousColumn] = (demographicsCohortsDf[continuousColumn] - mean)/std

        return demographicsCohortsDf

    # Parameters: diagnosisCohortsCleanedDf, dataframe, diagnosisDepthLevel3 df with only ICD-9 codes; categoricalColumns,
    #   list, list of categorical columns in df as strings.
    # Exception: None
    # Purpose: one-hot encode categorical columns (aggregate) and normalize continuous columns
    # Return: a featurized diagnosisDepthLevel3
    def featurizeAggregateDiagnosis(self, diagnosisCohortsCleanedDf, categoricalColumns=["dx"]):

        # subset
        diagnosisCohortsCleanedSubsetDf = diagnosisCohortsCleanedDf.loc[:, categoricalColumns + [self.patientID]]

        # one-hot encode
        for categoryColumn in categoricalColumns:
            one_hot = pd.get_dummies(diagnosisCohortsCleanedSubsetDf[categoryColumn])
            one_hot = one_hot.add_prefix(categoryColumn + "_")
            # one_hot = one_hot.drop(one_hot.columns[-1], axis=1)
            diagnosisCohortsCleanedSubsetDf = diagnosisCohortsCleanedSubsetDf.drop(categoryColumn, axis=1)
            diagnosisCohortsCleanedSubsetDf = diagnosisCohortsCleanedSubsetDf.join(one_hot)
        
        # groupby by patientID
        diagnosisCohortsCleanedSubsetDf = diagnosisCohortsCleanedSubsetDf.groupby(self.patientID).any()
        diagnosisCohortsCleanedSubsetDf = diagnosisCohortsCleanedSubsetDf*1
        diagnosisCohortsCleanedSubsetDf = diagnosisCohortsCleanedSubsetDf.reset_index()

        return diagnosisCohortsCleanedSubsetDf
    
    # Parameteres: prescriptionCohortsCleanedDf, dataframe, prescription df; categoricalColumns, list, list of
    #   categorical columns in df as strings.
    # Exception: None
    # Purpose: one-hot encode categorical columns (aggregate) and normalize continuous columns
    # Return: a featurized prescription
    def featurizeAggregatePrescription(self, prescriptionCohortsCleanedDf, categoricalColumns=["metamap_name"]):

        # subset
        prescriptionCohortsCleanedSubsetDf = prescriptionCohortsCleanedDf.loc[:, categoricalColumns + [self.patientID]]

        # one-hot encode
        for categoryColumn in categoricalColumns:
            one_hot = pd.get_dummies(prescriptionCohortsCleanedSubsetDf[categoryColumn])
            one_hot = one_hot.add_prefix(categoryColumn + "_")
            # one_hot = one_hot.drop(one_hot.columns[-1], axis=1)
            prescriptionCohortsCleanedSubsetDf = prescriptionCohortsCleanedSubsetDf.drop(categoryColumn, axis=1)
            prescriptionCohortsCleanedSubsetDf = prescriptionCohortsCleanedSubsetDf.join(one_hot)
        
        # groupby by patientID
        prescriptionCohortsCleanedSubsetDf = prescriptionCohortsCleanedSubsetDf.groupby(self.patientID).any()
        prescriptionCohortsCleanedSubsetDf = prescriptionCohortsCleanedSubsetDf*1
        prescriptionCohortsCleanedSubsetDf = prescriptionCohortsCleanedSubsetDf.reset_index()
        
        return prescriptionCohortsCleanedSubsetDf
    
    # Parameteres: procedureCohortsCleanedDf, dataframe, procedure df; categoricalColumns, list, list of
    #   categorical columns in df as strings.
    # Exception: None
    # Purpose: one-hot encode categorical columns (aggregate) and normalize continuous columns
    # Return: a featurized procedure df
    def featurizeAggregateProcedure(self, procedureCohortsCleanedDf, categoricalColumns=["dx"]):

        # subset
        procedureCohortsCleanedSubsetDf = procedureCohortsCleanedDf.loc[:, categoricalColumns + [self.patientID]]

        # one-hot encode
        for categoryColumn in categoricalColumns:
            one_hot = pd.get_dummies(procedureCohortsCleanedSubsetDf[categoryColumn])
            one_hot = one_hot.add_prefix(categoryColumn + "_")
            # one_hot = one_hot.drop(one_hot.columns[-1], axis=1)
            procedureCohortsCleanedSubsetDf = procedureCohortsCleanedSubsetDf.drop(categoryColumn, axis=1)
            procedureCohortsCleanedSubsetDf = procedureCohortsCleanedSubsetDf.join(one_hot)  
    
        # groupby by patientID
        procedureCohortsCleanedSubsetDf = procedureCohortsCleanedSubsetDf.groupby(self.patientID).any()
        procedureCohortsCleanedSubsetDf = procedureCohortsCleanedSubsetDf*1
        procedureCohortsCleanedSubsetDf = procedureCohortsCleanedSubsetDf.reset_index()

        return procedureCohortsCleanedSubsetDf       

    # Parameters: df, dataframe; textColumn, string, text column; minFreq, float, minimum doc freq; maxFreq, float, maximum doc freq.
    # Exception: None
    # Purpose: get n-grams (uni, bi-, and trigrams) and muti-hot encode each row
    # Return: a featurized report df
    def featurizedAggregateReports(self, df, textColumn="text", minFreq=0.005, maxFreq=0.95):
    
        vectorizer = CountVectorizer(ngram_range=(1,3), max_df=maxFreq, min_df=minFreq, binary=True)
        X = vectorizer.fit_transform(df[textColumn].tolist())
        ngramDf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        ngramDf[self.patientID] = df[self.patientID].tolist()
        return(ngramDf)

    # Parameters:
    #   df (dataframe): the table of information
    #   patientColumn (str): patientID column
    #   minFreq (float): minimum frequency across rows
    #   maxFreq (float): maximum frequency across rows
    # Exception: None
    # Purpose: filter for columns whose frequency across rows is greater than or equal to minFreq and less than or equal
    #   to maxFreq
    # Return: a filtered dataframe
    def featureFrequencyFiltration(self, df, patientColumn="patientID", minFreq=0.005, maxFreq=0.95):

        dfPrev = df.loc[:, df.columns != patientColumn].sum(axis=0)/len(df)
        dfFiltered = dfPrev[(dfPrev >= minFreq) & (dfPrev <= maxFreq)]
        columns = dfFiltered.index.tolist()
        columns.append(patientColumn)
        dfSubset = df.loc[:, columns]

        return(dfSubset)

    # Parameters:
    #   df (dataframe): dataframe of interest
    #   dayColumn (str): day column in df
    #   binSize (int): size of a single bin (in days)
    # Exception: binSize can't exceeds the total # of unique days
    # Purpose/Return: Map encounters to their bin based on the min value and then return a dataframe with the bin column
    def encountersToBins(self, df, dayColumn="dxDay", binSize=30):

        df["bin"] = np.floor((df.loc[:, dayColumn] - min(df.loc[:, dayColumn]))/binSize)
        df["bin"] = df["bin"].astype(int)

        return(df)

    # Parameters:
    #   df (dataframe): temporal df
    #   patientID (str): , patient ID column
    #   feature (str): feature column
    #   day (str), time column
    # Exception: None
    # Purpose: Feature temporal dataframe by patient and day and then pad if necessary
    # Return: Featurized temporal datafame
    def featurizeTemporal(self, df, patientID="patientID", feature="dx", day="bin"):

        # subset
        df = df.loc[:, [patientID, feature, day]]

        # identify max time stamps
        maxBins = len(df[day].unique())
        uniqueBins = set(df[day])

        # pivot
        dfPivot = pd.crosstab(index=[df[patientID], df[day]], columns=df[feature], normalize=False).reset_index()

        print("Pivot Completed")

        # identify patients that need padding
        dfPatients = self.getPatientsForPadding(dfPivot, maxBins, patientID, day)

        # add padding
        if (len(dfPatients) > 0):
            padding = self.addPaddingRows(dfPivot, dfPatients, uniqueBins=uniqueBins,
                                          day=day)
            dfPivot = pd.concat([dfPivot, padding], axis=0)

        print("Padding Completed")

        dfPivot = dfPivot.sort_values(by=[patientID, day], ascending=[True, True])

        return (dfPivot)

    # Parameters:
    #   df (dataframe): dataframe of interest
    #   maxBins (int): total # of bins
    #   patientID (str): patient ID column in df
    #   day (str): bin column of dataframe
    # Exception: None
    # Purpose: Identify patients thar require padding
    # Return: list of patientIDs
    def getPatientsForPadding(self, df, maxBins, patientID="patientID", day="timeStamps"):

        # get total encounters per patinet
        dfUnique = df.loc[:, [patientID, day]].drop_duplicates()
        patientIDCounts = dfUnique[patientID].value_counts()

        # identify patients that need padding
        patientIDCounts = patientIDCounts[patientIDCounts < maxBins]
        patientIDCounts = patientIDCounts.reset_index()
        patients = patientIDCounts["index"].tolist()

        return (patients)

    # Parameters:
    #   df (dataframe): dataframe of interest
    #   uniqueBins (set): all the unique bins in df
    #   patientID (str): patient ID column in df
    #   day (str): bin column in df
    # Exception: None
    # Purpose: Pad patients with 0's, so that # of encounters matches across patients
    # Return: df of padding
    def addPaddingRows(self, df, patients, uniqueBins, patientID="patientID", day="bins"):

        dfList = list()
        for patient in patients:
            output = self.addPaddingRowsSinglePatient(df, patient, uniqueBins, patientID, day)
            dfList.append(output)
        allRows = pd.concat(dfList)
        return (allRows)

    # Parameters:
    #   df (dataframe): dataframe of interest
    #   patient (int): patient id
    #   uniqueBins (set): unique set of bins in df
    #   patientID (str): patient ID column in df
    #   day (str): bin column in df
    # Exception: None
    # Purpose: Pad patient with 0's, so that # of encounters matches across patients
    # Return: df of padding for a single patient
    def addPaddingRowsSinglePatient(self, df, patient, uniqueBins, patientID="patientID", day="bins"):

        # get patient's info
        dfSubset = df.loc[df[patientID] == patient]

        # get patient's unique bins
        patientUniqueBins = set(dfSubset[day])

        # get difference
        difference = uniqueBins.difference(patientUniqueBins)

        paddingRows = pd.DataFrame(columns=df.columns)
        paddingRows[day] = list(difference)
        paddingRows[patientID] = np.repeat(patient, len(difference))
        paddingRows.fillna(0, inplace=True)

        return (paddingRows)

    # Parameters:
    #   reports (dataframe): index reports dataframe
    #   word2vecDir (str): directory path to word2vec model
    #   textColumn (str): text column of reports
    # Exception: None
    # Purpose: Convert text into a format that the deep learning can understand
    # Return: the original dataframe with a new column "indexedText" that has the list of indices from the word2vec
    #   model.
    def cnnSetup(self, reports, word2vecDir, textColumn="text"):

        reportsLens = [len(nltk.tokenize.word_tokenize(x)) for x in reports[textColumn].tolist()]
        seqLength = np.max(reportsLens).astype(int)
        print("Max Sequence Length: " + str(seqLength))
        word2vecModel = Word2Vec.load(word2vecDir)
        pretrainedWords = []
        for word in word2vecModel.wv.vocab:
            pretrainedWords.append(word)
        reportsList = reports.text.values.tolist()
        tokenizedReportsList = self.tokenizeAllReports(word2vecModel.wv, reportsList)
        print("Tokenized Reports")
        reportsFeatures = self.padTrimReports(tokenizedReportsList, seqLength)
        print("Padded and Trimmed Tokenized Reports")
        reports["indexedText"] = pd.Series(reportsFeatures.tolist())
        return(reports)

    # Parameters:
    #   embedLookup (word2vec obj): word2vec object for extracting the indices
    #   reports (list): list of report texts
    # Exception: None
    # Purpose: Convert text in reports to indices from word2vec model
    # Return: list of lists of indices from word2vec model
    def tokenizeAllReports(self, embedLookup, reports):
        reportsWords = [nltk.tokenize.word_tokenize(report) for report in reports]
        tokenizedReports = []
        for report in reportsWords:
            ints = []
            for word in report:
                try:
                    idx = embedLookup.vocab[word].index
                except:
                    idx = 0
                ints.append(idx)
            tokenizedReports.append(ints)
        return(tokenizedReports)

    # Parameters:
    #   tokenizedReportsList (list of lists): list of lists of indices from word2vec model
    #   seqLength (int): sequence of words to use
    # Exception: None
    # Purpose: Pad and trim reports based on the sequence length
    # Return: array of padded and trimmed reports
    def padTrimReports(self, tokenizedReportsList, seqLength):
        features = np.zeros((len(tokenizedReportsList), seqLength), dtype=int)
        for i, row in enumerate(tokenizedReportsList):
            sequenceWords = np.array(row)[:seqLength]
            # adds the sequence to the start of each row
            features[i, 0:len(sequenceWords)] = sequenceWords
        return(features)




