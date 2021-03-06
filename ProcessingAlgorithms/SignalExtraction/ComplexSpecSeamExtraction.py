import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.spatial import distance
from spectrogram import Spectrogram

def seamExtraction(SpectrogramObject:Spectrogram, startTime:int, stopTime:int, width:int, bottomIndex:int=0, topIndex:int=None, order:int = None, verbose:bool = False, theta = None):
    """
        Input:
            intensityMatrix: a 2d array [velocity][time] and each cell 
                corresponds to the intensity at that point.
            startTime: index corresponding to the start time in the intensity 
                matrix of the signal being considered.
            stopTime: index corresponding to the stop time in the intensity
                matrix of the signal being considered.
            width:
                How wide of a window that is being considered at
                    each time step. The window will be split evenly up and down
            bottomIndex: Upper bound on the velocity. 
                Assumed to be a valid index for sgram's intensity array.
                Defaults to zero
            order: the order of the minkowski distance function that will be minimized 
                to determine the most well connected signal. 
                Defaults to 2:
                    minimizing the Euclidean distance between neighboring pixels.
            verbose: Boolean that indicates if you want there to be debugging print statements.
                Defaults to False:
            theta: The relative importance of the phase array to the amplitude array if applicable.
                It should be between 0 and 1.
                
    Output:
        list of indices that should be plotted as the signal as an array of indices
        indicating the intensity values that correspond to the most 
        connected signal as defined by the metric.
    """

    if stopTime <= startTime:
        raise ValueError("Stop Time is assumed to be greater than start time.")

    else:
        try:
            theta = float(theta)
        except TypeError:
            raise TypeError("Theta needs to be able to be cast to a float.")

        if 0 > theta:
            theta = np.exp(theta) # This will get between 0 and 1.
        if theta > 1:
            theta = np.exp(-1*theta) # To get between 0 and 1.
        if width % 2 == 1:
            width += 1

        tableHeight = stopTime - startTime + 1

        velocities = np.zeros(tableHeight, dtype = np.int64)
        # This will hold the answers that we are looking for to return. It is a list of indices.

        halfFan = width//2

        bottomDP = bottomIndex + 1 


        if topIndex == None:
            topIndex = SpectrogramObject.velocity.shape[0] # I just want the maximum velocity index
        elif topIndex <= bottomIndex:
            topIndex = SpectrogramObject.velocity.shape[0]



        t, vel, real_raw_vals, ovals  = SpectrogramObject.slice((SpectrogramObject.time[startTime], SpectrogramObject.time[stopTime]), (SpectrogramObject.velocity[bottomDP], SpectrogramObject.velocity[topIndex]))

        amplitudeArray = [] # This is will be the magnitude of the complex data.
        phaseArray = []   # This will be the phase of the complex data.

        if SpectrogramObject.computeMode == "complex":
            # We have the complex data.
            amplitudeArray = real_raw_vals # It will be the appropriate item in this case.
            phaseArray = np.arctan2(np.real(ovals)/np.imag(ovals))
        elif SpectrogramObject.computeMode == "psd" or SpectrogramObject.computeMode == "magnitude":
            amplitudeArray = real_raw_vals # This is all that we will have as we cannot get the phase info.
        elif SpectrogramObject.computeMode == "angle" or SpectrogramObject.computeMode == "phase":
            phaseArray = ovals

        if verbose:
            print("new time shape", t.shape, "vel shape:", vel.shape)
            print("Before the transpose: real_raw_vals.shape", real_raw_vals.shape)      
            print("real_raw_vals.shape = ",real_raw_vals.shape)
            print("bottomDP", bottomDP)
            print()
            print("t2-t1", stopTime-startTime)
            print("(t2-t1)*halfan", (stopTime-startTime)*halfFan)
            print()
            print(topIndex)
            print(bottomIndex+1)
            print()
        
        amplitudeArray = np.transpose(amplitudeArray)
        phaseArray = np.transpose(phaseArray)
        tableHeight, tableWidth = np.transpose(real_raw_vals).shape
        DPTable = np.zeros((tableHeight, tableWidth))

        parentTable = np.zeros(DPTable.shape, dtype = np.int64)

        for timeIndex in range(2, tableHeight + 1):
            dpTime = tableHeight - timeIndex
            for velocityIndex in range(tableWidth):
                bestSoFar = np.Infinity
                bestPointer = None
                for testIndex in range(-halfFan, halfFan+1): # The + 1 is so that you get a balanced window.
                    if velocityIndex + testIndex >= 0 and velocityIndex + testIndex < tableWidth:
                        # then we have a valid index to test from.
                        ampAddition = 0
                        phaseAddition = 0
                        if len(amplitudeArray) != 0: # Then it has been initialized.
                            ampAddition = np.power(np.abs(amplitudeArray[dpTime+1][testIndex+velocityIndex]-amplitudeArray[dpTime][velocityIndex]), order)
                        if len(phaseArray) != 0: # Then it has been initialized.
                            phaseAddition = np.power(np.abs(phaseArray[dpTime+1][testIndex+velocityIndex]-phaseArray[dpTime][velocityIndex]), order)

                        current = (1-theta)*ampAddition + theta*phaseAddition + DPTable[dpTime+1][velocityIndex+testIndex]
                        
                        if current < bestSoFar:
                            bestSoFar = current
                            bestPointer = velocityIndex + testIndex
                            if verbose:
                                print("The bestValue has been updated for time", t[dpTime], "and velocity", vel[velocityIndex], \
                                    "to", bestSoFar, "with a bestPointer of ", vel[velocityIndex+testIndex], "at the next timeslice.")                          
                DPTable[dpTime][velocityIndex] = bestSoFar
                parentTable[dpTime][velocityIndex] = bestPointer

        # Now for the reconstruction.
        currentPointer = np.argmin(DPTable[0])
        
        if verbose:
            print("The value of the current pointer is", currentPointer)
            print("The minimum cost is", DPTable[0][currentPointer])

        velocities = reconstruction(parentTable, currentPointer, bottomDP)

        return velocities, parentTable, DPTable, bottomDP, topIndex


def reconstruction(parentTable, startPoint, bottomDP, verbose:bool =False):
    tableHeight = int(parentTable.shape[0])
    velocities = np.zeros(tableHeight, dtype = np.int64)
    for timeIndex in range(tableHeight):
        velocities[timeIndex] = startPoint + bottomDP
        if verbose:
            print(timeIndex)
            print("myStart", startPoint)
        startPoint = parentTable[timeIndex][startPoint]

    return velocities

def normalize(timeVelocityIntensity):
    """
        Time Vs Velocity intensity values.

        Subtract out the min at each time step. Then, find
        the maximum at that time step. Normalize the intensity
        at velocity along this time step by the maximum intensity
        at this time step. Return the normalized array.
    """
    newArray = np.zeros(timeVelocityIntensity.shape, dtype = np.float32)
    for timeInd in range(timeVelocityIntensity.shape[0]):
        minValue = np.min(timeVelocityIntensity[timeInd])
        timeVelocityIntensity[timeInd] += -1*minValue
        maxValue = np.max(timeVelocityIntensity[timeInd])
        for velInd in range(timeVelocityIntensity.shape[1]):
            curr = timeVelocityIntensity[timeInd][velInd]
            newValue = curr/maxValue
            newArray[timeInd][velInd] = newValue

    return newArray

def mainTest(SpectrogramObject:Spectrogram, startTime:int, stopTime:int, bottomIndex:int=0, topIndex:int=None, vStartInd: int=None,verbose:bool = False):
    widths = [1,3,5,11,21]
    orders = [1,2,10]
    precision = 10
    thetas = np.arange(precision+1)/precision # Get in the range of zero to one inclusive.
    seam = []

    velocities = SpectrogramObject.velocity
    time = SpectrogramObject.time

    signalData = time*1e6

    headers = ["Time ($\mu$s)"]

    basefilePath = "../DocumentationImages/DP/ComplexSpectra/"
    digfileUsed = SpectrogramObject.data.filename
    for width in widths:
        for order in orders:
            for theta in thetas:
                hyperParamNames =  "w " + str(width) + " ord " + str(order) + " Minkowski dist theta " + str(theta).replace(".", "_")
                signal, p_table, dp_table, botVel, topVel = seamExtraction(SpectrogramObject, startTime, stopTime, width, bottomIndex, topIndex, order, theta=theta)

                fname = "DP_Cost_Table time by vel " +hyperParamNames + ".csv"
                filename = basefilePath + fname
                np.savetxt(filename,dp_table,delimiter=",")

                fname = "Parent_Table time by vel " + hyperParamNames + ".csv"
                filename = basefilePath + fname
                np.savetxt(filename,p_table,delimiter=",")

                fig = plt.figure(num=1)
                plt.plot(velocities[botVel:topVel+1], dp_table[0])
                plt.title("Total Minkowski" + str(order) +" Order Cost diagram with a window size of " +str(width) +" PercPhase:" + str(theta))
                plt.xlabel("Starting velocity of the trace (m/s)")
                plt.ylabel("Minimum value of the sum ($\theta$|$\phi_i$ - $\phi_{i-1}$|^" + str(order) + "(1-$\theta$)|$Amp_i$ - $Amp_{i-1}$|^" + str(order) +") along the path")
                # manager = plt.get_current_fig_manager()
                # manager.window.Maximized()
                if verbose:
                    fig.show()
                    print("Here is a graph of the signal trace across time")       
                # fig.show()
                    
                extension = "svg"
                fname = "DP_Start_Cost " + hyperParamNames + extension
                filename = basefilePath + fname
                fig.savefig(filename, bbox_inches = "tight")

                fig2 = plt.figure(num = 2, figsize=(10, 6))
                ax = fig2.add_subplot(1,1,1)
                fig2Axes = fig2.axes[0]

                print(type(fig2Axes))
                print(fig2Axes)
                SpectrogramObject.plot(axes=fig2Axes)
                plt.xlim((SpectrogramObject.time[startTime], SpectrogramObject.time[stopTime]))

                fig2Axes.plot(time[startTime: stopTime+1], velocities[signal], 'b-', alpha = 0.4, label="Minimum Cost")
                if vStartInd != None:
                       # Compute the signal seam for assuming this is the start point.
                    seam = reconstruction(p_table, vStartInd-botVel, botVel)
                    fig2Axes.plot(time[startTime: stopTime+1], velocities[seam], 'r--', alpha = 0.4, label="Expected Start Point")
                fig2Axes.set_title("Velocity as a function of time for the minimum cost seam with Minkowski" + str(order)+ " and a window size of " + str(width) + " raw")
                fig2Axes.legend()

                # manager = plt.get_current_fig_manager()
                # manager.window.showMaximized()
                if verbose:
                    fig2.show()
                fname = "Overlay Spectra " + hyperParamNames + extension
                filename = basefilePath + fname
                plt.savefig(filename, bbox_inches = "tight")

                # Build the reconstruction of every possible starting velocity. Then, save them in a csv file.
                extension = "csv"
                fname = "Velocity Traces " + hyperParamNames + extension
                filename = basefilePath + fname

                for velInd in range(0, topVel-botVel+1):
                    trace = reconstruction(p_table, velInd, botVel)
                    header = "Starting Velocity " + str(velocities[velInd+botVel]) + "(m/s)"
                    headers.append(header) 
                    meaured = velocities[trace]
                    signalData = np.hstack((signalData, meaured))
                signalData = signalData.reshape((topVel-botVel + 2, len(time))).transpose() # I want each column to be a velocity trace.
                # np.savetxt(filename, signalData, delimiter=",")
                df = pd.DataFrame(data=signalData, index=time*1e6, columns=headers)
                df.to_csv(filename)

                print("Completed the documentation for ", hyperParamNames)

def documentationMain():
    # Set up

    MySpect, sTime, eTime, botInd, topVelInd, visSigIndex = setupForDocumentation()

    # Now document.
    mainTest(MySpect, sTime, eTime, botInd, topVelInd, visSigIndex)


def setupForDocumentation():
    # Set up
    filename = "../dig/CH_4_009.dig"
    t1 = 14.2389/1e6
    t2 = 31.796/1e6
    MySpect = Spectrogram(filename, mode = "complex")
    sTime = MySpect._time_to_index(t1)
    eTime = MySpect._time_to_index(t2)

    bottomVel = 1906.38
    botInd = MySpect._velocity_to_index(bottomVel)

    topVel = 5000
    topVelInd = MySpect._velocity_to_index(topVel)

    visualSignalStart = 2652.21
    visSigIndex = MySpect._velocity_to_index(visualSignalStart)

    return MySpect, sTime, eTime, botInd, topVelInd, visSigIndex