import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm # For printing a timing information to see how far you have gotten.


from spectrogram import Spectrogram




def saveBaselineIntensityImages(files, saveLoc:str = None, imageExt:str = "png"):
    """
    Save a image file of the baseline as a function of time to the folder specified by saveLoc.
    """
    if saveLoc == None:
        saveLoc = r"../baselineIntensityMaps"
    
    for i in tqdm.trange(len(files)):
        filename = f"../dig/{files[i]}"
        MySpect = Spectrogram(filename)
        peaks, _, heights = baselines.baselines_by_squash(MySpect)

        plt.plot(np.array(MySpect.time)*1e6, MySpect.intensity[MySpect._velocity_to_index(peaks[0])])
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel("Intensity (db)")
        plt.title(f"{MySpect.data.filename}")
        path = os.path.join(saveLoc, files[i].replace(".dig", "") + f"_baselineIntensity.{imageExt}")
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])

        plt.savefig(path)

        # Reset the state for the next data file.
        plt.clf()
        del MySpect

def baselineTracking(spectrogram:Spectrogram, baselineVel, changeThreshold, skipUntilTime:float=12e-6):
    """
        Return the first time in microseconds that baseline's intensity value changes outside the changeThreshold.
        Use the average baseline value as the estimate
    """

    baselineInd = spectrogram._velocity_to_index(baselineVel)
    runningAverage = spectrogram.intensity[baselineInd][spectrogram._time_to_index(spectrogram.t_start + skipUntilTime)]
    
    startInd = spectrogram._time_to_index(spectrogram.t_start + skipUntilTime)
    for ind in range(startInd, spectrogram.intensity.shape[-1]):
        currentIntensity = spectrogram.intensity[baselineInd][ind]
        if (np.abs(currentIntensity) >= np.abs((1+changeThreshold) * runningAverage)) or (np.abs(currentIntensity) <= np.abs((1-changeThreshold) * runningAverage)):
            # print(f"{changeThreshold}: {spectrogram.time[ind]*1e6}")
            return spectrogram.time[ind]*1e6
        runningAverage += 1/(ind+1-startInd) * (currentIntensity - runningAverage)
    
    return spectrogram.time[ind]*1e6


def runExperiment(trainingFilePath, thresholds:list, skipUntilTimes:list = []):
    currLocalDir = os.getcwd()
    os.chdir(os.path.split(__file__)[0])
    from baselines import baselines_by_squash
    os.chdir(currLocalDir)

    data = pd.read_excel(trainingFilePath)

    data = data.dropna() # Ignore the samples that we do not have the full data for.
    data.reset_index() # Reset so that it can be easily indexed.

    files = data["Filename"].to_list()
    bestGuessTimes = (data[data.columns[1]]*1e6).to_list()
    bestGuessVels = data[data.columns[-1]].to_list() # This will be the same times for the probe destruction dataset. It is generally ignored currently.


    d = {"Filename": files}

    if skipUntilTimes == []:
        skipUntilTimes = [12e-6]

    for i in range(len(thresholds)):
        d[f"threshold {thresholds[i]}"] = np.zeros((len(skipUntilTimes), len(files)))
        d[f"threshold {thresholds[i]} error"] = np.zeros((len(skipUntilTimes), len(files)))

    print("Running the experiment")

    for i in tqdm.trange(len(files)):
        filename = os.path.join(os.path.join("..", "dig") , f"{files[i]}")
        MySpect = Spectrogram(filename)
        peaks, _, heights = baselines.baselines_by_squash(MySpect)


        for thres in thresholds:
            
            for skipTimeInd in range(len(skipUntilTimes)):
                skipTime = skipUntilTimes[skipTimeInd]
                timeEstimate = baselineTracking(MySpect, peaks[0], thres, skipTime)

                d[f"threshold {thres}"][skipTimeInd][i] = timeEstimate
                d[f"threshold {thres} error"][skipTimeInd][i] = bestGuessTimes[i] - timeEstimate

    # Compute summary statistics
    summaryStatistics = np.zeros((len(thresholds), len(skipUntilTimes), 5))
    stats = ["Avg", "Std", "Median", "Max", "Min"]

    print(f"Computing the summary statistics: [{stats}]")

    for i in tqdm.trange(len(thresholds)):
        for j in range(len(skipUntilTimes)):
            q = np.power(d[f"threshold {thresholds[i]} error"][j],2)
            summaryStatistics[i][j][0] = np.mean(q)
            summaryStatistics[i][j][1] = np.std(q)
            summaryStatistics[i][j][2] = np.median(q)
            summaryStatistics[i][j][3] = np.max(q)
            summaryStatistics[i][j][4] = np.min(q)



    for i in tqdm.trange(len(stats)):
        fig = plt.figure()
        ax = plt.gca()

        pcm = ax.pcolormesh(np.array(skipUntilTimes)*1e6, thresholds, summaryStatistics[:,:,i])
        plt.title(f"{stats[i]} L2 error over the files")
        plt.ylabel("Thresholds")
        plt.xlabel("Time that you skip at the beginning")
        plt.gcf().colorbar(pcm, ax=ax)
    
        plt.show() # Need this for Macs.
        
    summaryFolder = r"../ProbeDestructionEstimates"
    if not os.path.exists(summaryFolder):
        os.makedirs(summaryFolder)


    for i in range(len(thresholds)):
        q = pd.DataFrame(summaryStatistics[i])
        internal_a = str(thresholds[i]).replace(".", "_")
        saveSummaryFilename = f"summaryThresh{internal_a}.csv"
        q.to_csv(os.path.join(summaryFolder, saveSummaryFilename))


    return summaryStatistics, d

if __name__ == "__main__":

    currDir = os.getcwd()
    os.chdir(Spectrogram.root_code_dir())

    thresholds = np.linspace(0.3, 1)
    skipUntilTimes = [12e-6]
    saveLoc = r"../baselineIntensityMaps"
    experimentFileName = r"../Labels/EstimateOfJumpOffPositionTrain.xlsx"

    data = pd.read_excel(experimentFileName)

    data = data.dropna() # Ignore the samples that we do not have the full data for.
    data.reset_index() # Reset so that it can be easily indexed.

    files = data["Filename"].to_list()

    if False:
        saveBaselineIntensityImages(files, imageExt = "jpeg")

    runExperiment(experimentFileName, thresholds, skipUntilTimes)
    os.chdir(currDir)