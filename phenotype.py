# Author: Chethan Jujjavarapu
# Purpose: Create the different cohorts based on LSS, LDH, and Decompression

import pandas as pd


class Phenotype:

    # Parameters: 
    #   timeWindowLumbar, int, the upper limit of days to search for LSS/LDH codes 
    #   timeWindowDecompression, int, time window to search for decompression surgery
    #   totalIcdCodes, int, number of icd codes to consider for LSS/LDH
    # Exception: None
    # Purpose/Return: Creates the time windows and code lists to be used throughout th class.
    def __init__(self, timePointMaxLumbar=487, timeWindowDecompression=365, totalIcdCodes=2):
        # total months for lss/ldh
        self.timePointMaxLumbar = timePointMaxLumbar

        # total icd codes to consider for lss/ldh
        self.totalIcdCodes = totalIcdCodes

        # 9 months for decompression
        self.timeWindowDecompression = timeWindowDecompression

        # codes developed by Pradeep Suri
        self.lssCodes = ['344.6',
                        '344.60',
                        '344.61',
                        '721.4',
                        '721.42',
                        '724',
                        '724',
                        '724.02',
                        '724.03',
                        '724.09',
                        'G83.4',
                        'M47.15',
                        'M47.16',
                        'M48.05',
                        'M48.06',
                        'M48.061',
                        'M48.062',
                        'M48.07',
                        'M48.08']
        self.ldhCodes=['344.6',
                     '344.60',
                     '344.61',
                     '353.4',
                     '355.0',
                     '721.4',
                     '721.42',
                     '722.1',
                     '722.10',
                     '724.3',
                     '724.4',
                     'G54.4',
                     'G57.0',
                     'G57.00',
                     'G57.01',
                     'G57.02',
                     'G83.4',
                     'M47.15',
                     'M47.16',
                     'M47.25',
                     'M47.26',
                     'M47.27',
                     'M47.28',
                     'M51.15',
                     'M51.16',
                     'M51.17',
                     'M54.18',
                     'M51.25',
                     'M51.26',
                     'M51.27',
                     'M54.10',
                     'M54.15',
                     'M54.16',
                     'M54.17',
                     'M54.18',
                     'M54.30',
                     'M54.31',
                     'M54.32',
                     'M54.4',
                     'M54.40',
                     'M54.41',
                     'M54.42']
        self.decompressionCodes = ['S9090',
                                     '246033',
                                     '245980',
                                     '63003',
                                     '63278',
                                     '0SB40ZZ',
                                     '63035',
                                     '245922',
                                     '009Y3ZZ',
                                     '0SB24ZZ',
                                     '0SB00ZX',
                                     '245964',
                                     '0S543ZZ',
                                     '63290',
                                     '0QW004Z',
                                     '63101',
                                     '707346',
                                     '63050',
                                     '63017',
                                     '63200',
                                     '63077',
                                     '63266',
                                     '63011',
                                     '63301',
                                     '00NY0ZZ',
                                     '80.59',
                                     '0S544ZZ',
                                     '245929',
                                     '63271',
                                     '0S523ZZ',
                                     '224089',
                                     '63307',
                                     '009Y40Z',
                                     '223900',
                                     '0SB33ZZ',
                                     '0QB10ZZ',
                                     '00NX0ZZ',
                                     '245927',
                                     '009Y0ZZ',
                                     '00NY3ZZ',
                                     '246793',
                                     '0RBA3ZZ',
                                     '245941',
                                     '01N83ZZ',
                                     '009T0ZZ',
                                     '245930',
                                     '0SB34ZZ',
                                     '0R5B0ZZ',
                                     '0PB40ZZ',
                                     '0QU03JZ',
                                     '63055',
                                     '63185',
                                     '63010',
                                     '0QS134Z',
                                     '0QB13ZX',
                                     '63172',
                                     '0SB20ZZ',
                                     '63306',
                                     '01NB0ZZ',
                                     '0SB03ZX',
                                     '63056',
                                     '0SB43ZX',
                                     '246792',
                                     '223899',
                                     '245987',
                                     '009U00Z',
                                     '63088',
                                     '63044',
                                     '245965',
                                     '63286',
                                     '0QU007Z',
                                     '0RB90ZZ',
                                     '0QS004Z',
                                     '222590',
                                     '224238',
                                     '22818',
                                     '0SB30ZZ',
                                     '009Y4ZZ',
                                     '224086',
                                     '251411',
                                     '253030',
                                     '80.51',
                                     '245948',
                                     '009T40Z',
                                     '0RBB3ZZ',
                                     '01NR0ZZ',
                                     '0QW034Z',
                                     '223880',
                                     '63308',
                                     '246035',
                                     '222572',
                                     '22102',
                                     '63066',
                                     '0275T',
                                     '0SB03ZZ',
                                     '63048',
                                     '226929',
                                     '245978',
                                     '0S520ZZ',
                                     '227539',
                                     '63046',
                                     '80.50',
                                     '222865',
                                     '245963',
                                     '245947',
                                     '245986',
                                     '756636',
                                     '0R5B3ZZ',
                                     '0QW104Z',
                                     '63051',
                                     '63302',
                                     '0QB03ZZ',
                                     '0SB23ZX',
                                     '0RBB0ZZ',
                                     '245931',
                                     '0QB00ZZ',
                                     '63102',
                                     '246790',
                                     '63091',
                                     '245998',
                                     '246791',
                                     '009T4ZZ',
                                     '224088',
                                     '245923',
                                     '227553',
                                     '009Y00Z',
                                     '63277',
                                     '00NY4ZZ',
                                     '0S524ZZ',
                                     '245939',
                                     '245944',
                                     '245983',
                                     '246797',
                                     '213730',
                                     '63005',
                                     '22819',
                                     '0SB00ZZ',
                                     '63276',
                                     '245936',
                                     '80.52',
                                     '222494',
                                     '231207',
                                     '245977',
                                     '009T3ZZ',
                                     '63057',
                                     '63283',
                                     '63078',
                                     '03.02',
                                     '245926',
                                     '63085',
                                     '63197',
                                     '245988',
                                     '245993',
                                     '80.53',
                                     '245991',
                                     '245937',
                                     '224131',
                                     '245935',
                                     '245989',
                                     '0S540ZZ',
                                     '246794',
                                     '0SB04ZZ',
                                     '63273',
                                     '63030',
                                     '0ST40ZZ',
                                     '00JU0ZZ',
                                     '63173',
                                     '246796',
                                     '245949',
                                     '0QU00JZ',
                                     '63012',
                                     'S2351',
                                     '63086',
                                     '63287',
                                     '251410',
                                     '63064',
                                     '63090',
                                     '63191',
                                     '63281',
                                     '63303',
                                     '224087',
                                     '03.09',
                                     '63016',
                                     '0R5B4ZZ',
                                     '0ST20ZZ',
                                     '63047',
                                     '63199',
                                     '03.01',
                                     '245994',
                                     '63267',
                                     '0PB43ZX',
                                     '246795',
                                     '009T30Z',
                                     '245999',
                                     '63103',
                                     '63170',
                                     '62380',
                                     '245945',
                                     '63282',
                                     '009T00Z',
                                     '00JV0ZZ',
                                     '245996',
                                     '63305',
                                     '63268',
                                     '80.54',
                                     '009U0ZZ',
                                     '0RBA0ZZ',
                                     '01N80ZZ',
                                     '0SC00ZZ',
                                     '63272',
                                     '0QB03ZX',
                                     '245946',
                                     '224928',
                                     '009Y30Z',
                                     '245925',
                                     '246789',
                                     '246000',
                                     '224170',
                                     '0SB23ZZ',
                                     '63087',
                                     '245940',
                                     '707347',
                                     '246034',
                                     '224929',
                                     '224085',
                                     '223901',
                                     '245997',
                                     '0RTB0ZZ',
                                     '0RBB3ZX',
                                     '0RB93ZX',
                                     '01NB4ZZ',
                                     '231208',
                                     '03.6',
                                     '227538',
                                     '0274T',
                                     '63042',
                                     '245942',
                                     '0RBA4ZZ',
                                     '0RBB4ZZ',
                                     '222573',
                                     '245938',
                                     'S2350',
                                     '0SB44ZZ',
                                     '63190',
                                     '0SB43ZZ',
                                     '63195']

    # Paramters: 
    #   diagnosisDf, dataframe, contains the diagnosisDepthLevel3 codes for each patient in the LIRE system
    #   procedureDf, dataframe, contains the procedure codes for each patient in the LIRE system
    #   patientID, string, patient ID column
    #   code, string, diagnosisDepthLevel3 code column
    #   dxDay, int, diagnosisDepthLevel3 day column
    #   prodDay, int, procedure day
    # Exceptions: None
    # Purpose/Return: Returns a dataframe that identifies the lss/ldh patients for positive and negative groups
    #   based on decompression. Includes the patient ID, diagnosisDepthLevel3 day of lss/ldh, procedure day for decompression,
    #   and group label (positive or negative).
    def cohortBuilder(self, diagnosisDf, procedureDf, patientID="patientID", code="dx", dxDay="dxDay",
                      procDay="procDay"):

        patientList = self.lssLdhBuilder(diagnosisDf, patientID, code, dxDay)
        print("Total LSS/LDH Patients: " + str(len(patientList)))
        cohorts = self.decompressionBuilder(patientList, procedureDf, patientID, code, dxDay, procDay)
        return(cohorts)

    # Parameters: 
    #   diagnosisDf, dataframe, contains the diagnosisDepthLevel3 codes for each patient in the LIRE system;
    #   patientID, string, patient ID column; 
    #   code, string, diagnosisDepthLevel3 code column;
    #   dxDay, int, diagnosisDepthLevel3 day column.
    # Exception: There are no exceptions.
    # Purpose/Return: Returns a dataframe that represents the patients with lss/ldh that had their second code within
    #   3 months after the index time point (i.e. 0 day).
    def lssLdhBuilder(self, diagnosisDf, patientID="patientID", code="dx", dxDay="dxDay"):

        # subset
        diagnosisDfSubset = diagnosisDf[(diagnosisDf[dxDay] < self.timePointMaxLumbar) &
                                        (diagnosisDf[code].isin(self.ldhCodes + self.lssCodes))]

        # get count of codes for each patient
        diagnosisDfSubsetCounts = diagnosisDfSubset[patientID].value_counts()

        # subset for patients with atleast 2 counts of these codes
        diagnosisPatientsList = diagnosisDfSubsetCounts[diagnosisDfSubsetCounts >= self.totalIcdCodes].index

        return(diagnosisPatientsList)

    # Parameters: 
    #   patientList, seroes, patients that have the required # of lss/ldh codes in the given time frame; 
    #   procedureDf, dataframe, contains the procedure code for each patient in the LIRE system;  
    #   patientID, string, patient ID column; 
    #   code, string, code column;
    #   dxDay, int, diagnosisDepthLevel3 day;
    #   procDay, int, procedure day.
    # Exception: if patientList is empty then method cannot proceed.
    # Purpose/Return: Returns a dataframe that contains lss/ldh patients in the positive group or negative group
    #   based on decompression surgery. Includes the patient ID, diagnosisDepthLevel3 day of lss/ldh, procedure day for
    #   decompression, and group label (positive or negative).
    def decompressionBuilder(self, patientList, procedureDf, patientID="patientID", code="dx", dxDay="dxDay",
                             procDay="procDay"):

        if patientList.shape[0] == 0:
            raise Exception("There are no patients with lss/ldh codes")

        # subset for patients with decompression
        procedureDfSubset = procedureDf.loc[procedureDf[code].isin(self.decompressionCodes), :].loc[:,[patientID, procDay]].sort_values(by=[patientID, procDay]).drop_duplicates()

        # get the first code
        patientEncounterDecompressionFirstTimePointTable = procedureDfSubset.groupby(patientID).nth(0).reset_index()

        # left join with lssLdhDf
        lssLdhDecompressionDf = pd.merge(pd.DataFrame({patientID:patientList}),
                                         patientEncounterDecompressionFirstTimePointTable,
                                         on=patientID,
                                         how="left")
        # get groups

        ## negative (no decompression surgery)
        negativeGroup = lssLdhDecompressionDf[lssLdhDecompressionDf[procDay].isna()]

        ## positive
        positiveGroup = lssLdhDecompressionDf[lssLdhDecompressionDf[procDay].notnull()]
        positiveGroup = positiveGroup[(positiveGroup[procDay] >= self.timePointMaxLumbar) &
                                      (positiveGroup[procDay] < self.timePointMaxLumbar + self.timeWindowDecompression)]

        ## combine
        negativeGroup["Group"] = "negative"
        positiveGroup["Group"] = "positive"
        print("Total LSS/LDH Patients with Decompression: " + str(len(positiveGroup)))
        print("Total LSS/LDH Patients with No Decompression: " + str(len(negativeGroup)))
        cohorts = pd.concat([negativeGroup, positiveGroup])
        cohorts["lumbarDay"] = self.timePointMaxLumbar

        return(cohorts)
