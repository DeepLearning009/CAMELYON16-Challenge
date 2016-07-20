'''
Created on 9 Jun 2016

@author: hjlin
'''
import os
import sys

def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR,"r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        Probs.append(float(elems[0]))
        Xcorr.append(int(elems[1]))
        Ycorr.append(int(elems[2]))
    return Probs, Xcorr, Ycorr

if __name__ == '__main__':
    result_folder = "./RefinedResults"
    overallCsvFileName = "ClassifyResults.csv"
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.csv')]
    
    newcsvPath = result_folder + "/" + overallCsvFileName
    file_csv = open(newcsvPath,'w')
    
    if not file_csv:
        print "Cannot open the file_csv %s for writing" %newcsvPath
    
    for case in result_file_list:
        print 'Evaluating Probability on Image:', case[0:-4]
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
        maxProb = max(Probs)
        print "Probability = %f" %maxProb
        file_csv.write( "%s,%f"%(case[0:-4],maxProb) +"\n")
    file_csv.close()
    
    print "Finish All Classification Evaluation"
    
