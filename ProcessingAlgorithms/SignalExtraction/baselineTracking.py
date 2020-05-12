import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm # For printing a timing information to see how far you have gotten.


from spectrogram import Spectrogram
from UI_Elements.plotter import COLORMAPS, DEFMAP


from spectrogram import Spectrogram
from ProcessingAlgorithms.SignalExtraction.baselines import baselines_by_squash


def saveBaselineIntensityImages(files, saveLoc: str = None, imageExt: str = "png"):
    """
    Save a image file of the baseline as a function of time to the folder specified by saveLoc.
    """
    if saveLoc == None:
        saveLoc = r"../baselineIntensityMaps"

    for i in tqdm.trange(len(files)):
        filename = f"../dig/{files[i]}"
        MySpect = Spectrogram(filename)
        peaks, _, heights = baselines_by_squash(MySpect)

        plt.plot(np.array(MySpect.time) * 1e6,
                 MySpect.intensity[MySpect._velocity_to_index(peaks[0])])
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
    runningAverage = spectrogram.intensity[baselineInd][spectrogram._time_to_index(
        spectrogram.t_start + skipUntilTime)]

    startInd = spectrogram._time_to_index(
        spectrogram.t_start + skipUntilTime)
    for ind in range(startInd, spectrogram.intensity.shape[-1]):
        currentIntensity = spectrogram.intensity[baselineInd][ind]
        if (np.abs(currentIntensity) >= np.abs((1 + changeThreshold) * runningAverage)) or (np.abs(currentIntensity) <= np.abs((1 - changeThreshold) * runningAverage)):
            # print(f"{changeThreshold}: {spectrogram.time[ind]*1e6}")
            return spectrogram.time[ind] * 1e6
        runningAverage += 1 / (ind + 1 - startInd) * \
            (currentIntensity - runningAverage)

    return spectrogram.time[ind] * 1e6


def baselineExperiment(baselineIntensity:np.array, thresholds:np.array, skipFirstXIndices:int = 0):

    numPassedThres = 0
    currTimeInd = 0
    outputTimeInds = np.zeros(len(thresholds), dtype = int)

    time_len = len(baselineIntensity)
    runningAverage = baselineIntensity[skipFirstXIndices]

    while currTimeInd + skipFirstXIndices < time_len:
        currentIntensity = baselineIntensity[currTimeInd+skipFirstXIndices]
        while (np.abs(currentIntensity) >= np.abs((1+thresholds[numPassedThres]) * runningAverage)) \
            or (np.abs(currentIntensity) <= np.abs((1-thresholds[numPassedThres]) * runningAverage)):
            
            outputTimeInds[numPassedThres] = int(currTimeInd + skipFirstXIndices)
            numPassedThres += 1
            if numPassedThres >= len(thresholds):
                return outputTimeInds # nothing left to check.
        runningAverage += 1 / (currTimeInd + 1) * (currentIntensity - runningAverage)
        currTimeInd += 1


    for k in range(numPassedThres, len(thresholds)):
        outputTimeInds[k] = time_len - 1

    return outputTimeInds

def runExperiment(trainingFilePath, thresholds:list, skipUntilTimes:list = [], cmap:str = DEFMAP, errorPower:int = 1):

    data = pd.read_excel(trainingFilePath)

    # Ignore the samples that we do not have the full data for.
    data = data.dropna()
    data.reset_index()  # Reset so that it can be easily indexed.

    files = data["Filename"].to_list()
    bestGuessTimes = (data[data.columns[1]] * 1e6).to_list()
    # This will be the same times for the probe destruction dataset. It is generally ignored currently.
    bestGuessVels = data[data.columns[-1]].to_list()

    d = {"Filename": files}

    if skipUntilTimes == []:
        skipUntilTimes = [12e-6]

    cmapAttempt = cmap
    if cmapAttempt in COLORMAPS.keys():
        cmapAttempt = COLORMAPS[cmapAttempt]

    for i in range(len(thresholds)):
        d[f"threshold {thresholds[i]}"] = np.zeros((len(skipUntilTimes), len(files)))
        d[f"threshold {thresholds[i]} error"] = np.zeros((len(skipUntilTimes), len(files)))

    print("Running the experiment")

    for i in tqdm.trange(len(files)):
        filename = os.path.join(os.path.join("..", "dig") , f"{files[i]}")
        MySpect = Spectrogram(filename,  overlap = 0)
        peaks, _, heights = baselines_by_squash(MySpect, min_percent = 0.75)
        baselineInd = MySpect._velocity_to_index(peaks[0])
        intensity = MySpect.intensity[baselineInd]

        for skipTimeInd in range(len(skipUntilTimes)):
            skipXInds = MySpect._time_to_index(skipUntilTimes[skipTimeInd])


            outputTimeInds = np.array(baselineExperiment(intensity, thresholds, skipXInds), dtype = int)

            for thresInd, thres in enumerate(thresholds):
                timeEstimate = MySpect.time[outputTimeInds[thresInd]]*1e6

                d[f"threshold {thres}"][skipTimeInd][i] = timeEstimate
                d[f"threshold {thres} error"][skipTimeInd][i] = bestGuessTimes[i] - timeEstimate


        del MySpect


    # Compute summary statistics
    summaryStatistics = np.zeros((len(thresholds), len(skipUntilTimes), 5))
    stats = ["Avg", "Std", "Median", "Max", "Min"]

    print(f"Computing the summary statistics: [{stats}]")

    for i in tqdm.trange(len(thresholds)):
        for j in range(len(skipUntilTimes)):
            q = np.power(d[f"threshold {thresholds[i]} error"][j], errorPower)
            summaryStatistics[i][j][0] = np.mean(q)
            summaryStatistics[i][j][1] = np.std(q)
            summaryStatistics[i][j][2] = np.median(q)
            summaryStatistics[i][j][3] = np.max(q)
            summaryStatistics[i][j][4] = np.min(q)

    for i in tqdm.trange(len(stats)):
        fig = plt.figure()
        ax = plt.gca()
        
        pcm = ax.pcolormesh(np.array(skipUntilTimes)*1e6, thresholds, summaryStatistics[:,:,i], cmap = cmapAttempt)
        plt.title(f"{stats[i]} Error over the Files Tested")
        plt.ylabel("Thresholds")
        plt.xlabel("Time that you skip at the beginning ($\mu$s)")
        plt.gcf().colorbar(pcm, ax=ax)

        plt.show()  # Need this for Macs.

    summaryFolder = r"../JumpOffTimeEstimates"
    if not os.path.exists(summaryFolder):
        os.makedirs(summaryFolder)

    for i in range(len(thresholds)):
        q = pd.DataFrame(summaryStatistics[i])
        internal_a = str(thresholds[i]).replace(".", "_")
        saveSummaryFilename = f"summaryThresh{internal_a}.csv"
        q.to_csv(os.path.join(summaryFolder, saveSummaryFilename))

    return summaryStatistics, d


def runAgainstTestSet(testSetFilename, guessAtOptimalThres, guessAtOptimalSkipUntil, errorPower:int = 1):

    data = pd.read_excel(testSetFilename)
    
    # Ignore the samples that we do not have the full data for.
    data = data.dropna()
    data.reset_index()  # Reset so that it can be easily indexed.

    files = data["Filename"].to_list()
    
    bestGuessTimes = (data[data.columns[1]] * 1e6).to_list()
    # This will be the same times for the probe destruction dataset. It is generally ignored currently.
    bestGuessVels = data[data.columns[-1]].to_list()

    d = {"Filename": files}


    d["ProbeDestructionEstimateTime"] = np.zeros(len(files))
    d["Error Versus Label"] = np.zeros(len(files))

    for i in tqdm.trange(len(files)):
        filename = os.path.join(os.path.join("..", "dig") , f"{files[i]}")
        MySpect = Spectrogram(filename,  overlap = 0)
        peaks, _, heights = baselines_by_squash(MySpect, min_percent = 0.75)
        timeEstimate = baselineTracking(MySpect, peaks[0], guessAtOptimalThres, guessAtOptimalSkipUntil)
        d["ProbeDestructionEstimateTime"][i] = timeEstimate
        d["Error Versus Label"][i] = np.power(bestGuessTimes[i] - timeEstimate, errorPower)

    localerrors = d["Error Versus Label"]

    print(f"The statistics for the test set specified by {testSetFilename}\n")
    print(f"Avg: {np.mean(localerrors)} Std: {np.std(localerrors)} Median: {np.median(localerrors)} Max: {np.max(localerrors)} Min: {np.min(localerrors)}")

    return d


if __name__ == "__main__":

    # thresholds = np.linspace(0.3, 1)
    # skipUntilTimes = [12e-6]
    # saveLoc = r"../baselineIntensityMaps"
    # experimentFileName = r"../Labels/JustGen3DataProbeDestructTrain.xlsx"

    # data = pd.read_excel(experimentFileName)

    # data = data.dropna() # Ignore the samples that we do not have the full data for.
    # data.reset_index() # Reset so that it can be easily indexed.

    # files = data["Filename"].to_list()

    # if False:
    #     saveBaselineIntensityImages(files, imageExt = "jpeg")

    # runExperiment(experimentFileName, thresholds, skipUntilTimes)
    pass
