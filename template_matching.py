#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: To automatically find desired regions in 
            an intensity matrix that match specified templates.
  Created: 1/28/2020
"""

import numpy as np
import pandas as pd
from spectrogram import Spectrogram
import peak_follower
import baselines as bls


from ImageProcessing.Templates.templates import *
# These are the templates that are in the folder ImageProcessing\Templates\templates.py
# How it is currently written will import everything from that file.

class Template:

    def __init__(self,
                 values:list = None,
                 vertical:bool = True
                 ):

        """
        Constructor for Template Objects.

        Inputs:
        -  values: the 1D or 2-dimensional array of weights for the template
        -  vertical: a boolean that specifies if the template is 1-d whether
                it is to be used vertically or horizontally. Defaults to vertical.

        Outputs:
        -  template object: This will be used to calculate a score for the
                                region it is placed on in the spectrogram.

        Observations:
            Using multiple templates for a combined score has proven to work 
            marginally better than a singular template. The template values
            are multiplied by their coresponding intensity values in the spectrogram,
            and then summed to get a total score for that region.

            You can make more templates as shown in the ImageProcessing\Templates\templates.py
            file.
        """
        self.values = np.array(values) if values != None else np.empty((0,0))

        self.width = None  # How wide.
        self.height = None # How tall.
        if len(self.values.shape) == 2:
            self.height, self.width = self.values.shape

        elif len(self.values.shape) == 1:
            if vertical:
                self.height = self.values.shape[0]
            else:
                self.width = self.values.shape[0]
        else:
            raise ValueError(f"It is assumed that the input template will be a 2-dimensional template. " \
        +"You input an array which has {len(self.values.shape)}-dimensions")




    # def calculate_score(self, intensities:list, velInd:int, timeInd:int):
    #     """
    #         Input:
    #             intensities: This is a 2-d matrix that I will assume has the appropriate shape
    #                 to match the template in the positive quadrant with
    #                 the origin at the index (velInd, timeInd).
    #             velInd: integer
    #             timeInd: integer
    #         Output:
    #             Compute the product sum of the intensities and the template
    #             starting at velInd and timeInd.
    #             returns - float
    #     """
    #     intensities = np.array(intensities) # convert the system to an numpy array so that you can slice it easily.

    #     regionIntensity = None

    #     if self.width == None:
    #         # This is a vertical template.
    #         regionIntensity = intensities[velInd:velInd+self.height+1,timeInd]
    #     elif self.height == None:
    #         # This is a horizontal template.
    #         regionIntensity = intensities[velInd,timeInd:timeInd+self.width+1]
    #     else:
    #         # This is the standard 2-d template.
    #         # I am assuming that the height will align with the
    #         # velocity and the width with the time axes.
    #         regionIntensity = intensities[velInd:velInd+self.height, timeInd:timeInd+self.width]
    #     return np.sum(self.values*regionIntensity)



def calculate_score(velo_index, template, intensities, time_index):
    """
    Returns the score for the template on a certain region in the
    spectrogram. The score is the sum of all products of the template 
    values and their corresponding intensity values. 

    Inputs:
      -  velo_index: the velocity index of the spectrogram that the template
                    will start at when computing the sum of all products. 
      -  template: a template object with values, a width, and a height. 
      -  intensities: a 2 dimensional array of values that are produced by a 
                    rolling fast fourier transform of voltage data. See
                    Spectrogram.py for more details. 
      - time_index: the time offset specified by the user to index into the 
                    2 dimensional array of intensities. 

    Outputs:
      -  template_sum: the sum of all inner products between intensities and 
                    template values. 
    """

    template_sum = 0        

    for values in template.values:

        for i, value in enumerate(values):

            template_sum += value * intensities[velo_index][time_index+i]

    return template_sum



def find_potential_baselines(sgram):
    """
    Returns a list of velocity values that correspond to the start of 
    a potential baseline in the spectrogram. The user will be asked to 
    verify that the baseline found is accurate, and indeed a baseline. 

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.

    Outputs:
      -  new_baselines: a list of velocity values that correspond to the start of 
                a potential baseline in the input spectrogram. 

    """
    #TODO use it with peak follower
    #TODO dot product between one dimensional vectors using .flatten
    #TODO expand to cover crossings and other phenomena

    peaks, dvs, heights = bls.baselines_by_squash(sgram) 

    df = pd.DataFrame(dict(peaks=peaks, heights=heights))

    new_df = df[df.heights > .13]

    baselines = []
    
    for index, rows in new_df.iterrows():         

        baselines.append(rows.peaks) 

    return baselines




def find_start_time(sgram):
    """
    Returns a potential jump off time for each baseline after 
    prompting the user for that input. 

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.

    Outputs:
      -  time_index: the start time index that can be used to find actual 
                times in the spectrogram.time array. 

    """

    time_index = None
    # start_time = 12 * 10**-6
    # time_index = sgram._time_to_index(start_time)

    ans = input("Where does the start begin? (in microseconds)\n")
    try:
        start_time = int(ans) * 10**-6
        # baseline_index = sgram._velocity_to_index(baseline)
        time_index = sgram._time_to_index(start_time)
    except:
        print("Input can not be converted to integer")
        return None

    return time_index



def find_regions(sgram, templates):
    """
    Returns a dictionary of dictionaries. The outermost dictionary keys are 
    time values. Those time values each have they're own dictionaries, with 
    velocity keys corresponding to the score calculated with that key's velocity
    as the starting position for each of the templates.

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.
      - templates: a list of template objects.
      - velo bounds: will keep the algorithm from searching outside of certain velocity ranges
      - time bounds: will keep the algorithm from searching outside of certain time ranges

    Outputs:
      -  all_scoures: a dictionary of dictionaries. Which contain the scores from the inputed 
                    templates and spectrogram.intensity matrix.  
    """

    baselines = find_potential_baselines(sgram)

    if len(baselines) > 1:

        baselines = sorted(baselines, reverse=True)

    print(baselines)



    time_index = find_start_time(sgram)

    time_max = sgram.intensity.shape[1]-1
    time_min = 0

    
    max_time = sgram.intensity.shape[1]-1

    if time_index is None:
        return

    upper_velo_index = sgram.intensity.shape[0]-1

    # print(time_max)
    # print("upper: ",upper_velo_index)
    # print("lower: ",lower_velo_index)
    # print(sgram.intensity)
    # print(template.values)
    # print(time_index)

    baseline_scores = {}

    for i, baseline in enumerate(baselines):

        if i != 0:
            upper_velo_index = sgram._velocity_to_index(baselines[i-1]-50.0)

        print("for baseline: ",upper_velo_index)

        lower_velo = baseline + 50.0
        lower_velo_index = sgram._velocity_to_index(lower_velo)

        baseline_scores[baseline] = {}
        # all_scores = {}

        if len(templates) > 1:
            width = templates[0].width
        else:
            width = templates.width

        start_index = time_index - (2*width)
        end_index = time_index + (2*width)

        if start_index < 0:
            start_index = 0
            end_index = width*4
        if end_index+width > max_time:
            end_index = max_time-width-1

        for time in range(start_index, end_index, 1):

            scores = {}

            for velocity_index in range(upper_velo_index, lower_velo_index, -1):

                score = 0

                for template in templates:

                    # score += template.calculate_score(sgram.intensity, velocity_index, i)
                    # print(score)
                    score += calculate_score(velocity_index, template, sgram.intensity, time)
                
                scores[velocity_index] = score
            
            baseline_scores[baseline][time] = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
            # all_scores[time] = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}


            # out = list(baseline_scores[baseline][time].items())[-5: -1]
            # for k, v in out:
        #         print("    time: ", sgram.time[time])
        #         print("velocity: ",sgram.velocity[k])
        #         print("   value: ", v, "\n")
        #     print()

        # print("\n\n")
                
    
    return baseline_scores, baselines
    return all_scores



def find_potenital_start_points(sgram, all_scores):
    """
    Returns a list of interesting coordinates in the spectrogram
    that can correspond to potential jumpoff points. These 
    points are likely near the actual starting 
    jumpoff point. 

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.
      - all_scores: a dictionary of dictionaries.

    Outputs:
      -  interesting_points: a list of (time, velocity) tuples.  
    """

    temp = []

    for time in all_scores.keys():

        keys = list(all_scores[time].keys())
        indicies = []

        for i in range(len(keys)-1, len(keys)-5, -1):
            tup = (time, keys[i])
            indicies.append(tup)

        # print(time,'\n')
        # print(all_scores[time][i])
        # print(indicies,'\n')

        for i in indicies:
            tup = (i, sgram.velocity[i[1]], all_scores[i[0]][i[1]])
            temp.append(tup)
    
    final = sorted(temp, key = lambda x: x[2], reverse=True)

    interesting_points = []

    total_score = 0
    for i in range(10):
        total_score += final[i][2]

    average_score = total_score/10
    print(average_score)

    for i in range(60):

        tup, velo, score = final[i]
        
        t, v = tup

        if score > average_score:
            interesting_points.append((sgram.time[t], velo))
            
    print(len(interesting_points))
    return interesting_points




if __name__ == '__main__':
    import os
    from digfile import DigFile

    path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs"
    os.chdir(path)
    df = DigFile('WHITE_CH3_SHOT.dig')


    sgram = Spectrogram(df, 0.0, 60.0e-6, form='power')

    find_potential_baselines(sgram)


    # template = Template(values=start_pattern)
    # template2 = Template(values=start_pattern2)
    # template3 = Template(values=start_pattern3)
    # template4 = Template(values=start_pattern4)
    template = Template(values=bigger_start_pattern)
    template2 = Template(values=bigger_start_pattern2)
    template3 = Template(values=bigger_start_pattern3)


    templates = [template, template2, template3]


    baseline_scores, baselines = find_regions(sgram, templates)

    for baseline in baselines:

        interesting_points = find_potenital_start_points(sgram, baseline_scores[baseline])

        total_velo = 0

        for i in interesting_points:

            time, velo = i

            total_velo += velo

        average_velo = total_velo / len(interesting_points)
        print(average_velo)


