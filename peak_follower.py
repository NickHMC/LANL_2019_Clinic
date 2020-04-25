#!/usr/bin/env python3
# coding:utf-8
"""
::
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
from follower import Follower
from ProcessingAlgorithms.BoundingAreaOfInterest.baselines import baselines_by_squash as bline
from ProcessingAlgorithms.Fitting.moving_average import moving_average


class PeakFollower(Follower):
    """
    A naive follower implementation that uses a local-region
    smoothing and then follows the local maximum.
    **Inputs to the constructor**
    - spectrogram: an instance of Spectrogram
    - start_point: (t, v), the coordinates at which to begin the search
                The time is in seconds. The velocity is in m/s.
    - span: (60) the number of pixels on either side of the starting value of v
      at which to search for a peak at the next time step.
    - smoothing: (4) the number of points on either side of a given velocity
      in a spectrum to average over to produce a smoothed representation of the
      spectrum prior to searching for a peak.
    - max_hop: (50) the largest change in v from the previous time step to consider
      as a continuation.
    """

    def __init__(self, spectrogram, start_point, span=80,
                 smoothing=4,  # average this many points on each side
                 max_hop=80,   # require peaks at successive time steps
                 # to be within this many velocity indices
                 debug = False # Do you want to print out debugging statements each loop?
                 ):
        super().__init__(spectrogram, start_point, span, debug=debug)
        self.smoothing = smoothing
        self.max_hop = max_hop
        peaks, dv, heights = bline(spectrogram)
        self.baselines = np.array(peaks)

        # It was passed in so we will assume that it is correct!
        self.results["velocities"].append(self.v_start)
        self.results["times"].append(self.t_start)
        self.results["time_index"].append(self.time_index)
        self.results['intensities'].append(self.spectrogram.intensity[self.spectrogram._velocity_to_index(self.v_start), self.time_index])
        self.results["velocity_index_spans"].append((self.spectrogram._velocity_to_index(self.v_start)-span, self.spectrogram._velocity_to_index(self.v_start) + span))

        self.time_index += 1
    def run(self, maxIter = 0):
        """
        Repeatedly call step until it fails.
        Returns the total intensity observed along the trajectory up to the number of iterations allowed by maxIter.
        If maxIter <= 0 then it will run until completion. 
        """
        while self.step(maxIter):
            pass
        eIter = len(self.results['intensities']) if maxIter <= 0 else maxIter
        return np.sum(self.results['intensities'][:eIter])
        

    def step(self, maxIter=0):
        """
        Attempt to fit a gaussian starting from the coeffients
        in the input parameter to intensities vs velocities.
        If coefficients is None or empty, guess reasonable values.
        The order of the parameters in the coefficients array is
        amplitude, center, width, and background.
        """
        velocities, intensities, p_start, p_end = self.data()

        if self.debug:
            print("The list of my times",self.results['times'])
            print("The list of my velocities", self.results['velocities'])
        
        if self.time_index >= self.spectrogram._time_to_index(self.spectrogram.probe_destruction_time):
            return False

        if maxIter > 0:
            if len(self.results['time_index']) >= maxIter:
                return False # We have done the maximum number of iterations.

        if self.smoothing:
            intensities = moving_average(intensities, n=self.smoothing)

        # generate an index array to sort the intensities (low to high)
        low_to_high = np.argsort(intensities)

        # remove any peaks that are too close to the baseline
        n = -1
        while True:
            mingap = np.min(
                np.abs(self.baselines - velocities[low_to_high[n]]))
            if mingap > 100:
                break
            n -= 1

        top = low_to_high[n]  # index of the tallest intensity peak
        v_high = velocities[top]

        # add this to our results
        self.results['velocity_index_spans'].append((p_start, p_end))
        self.results['time_index'].append(self.time_index)
        self.results['times'].append(
            self.spectrogram._point_to_time(self.time_index))
        self.results['velocities'].append(velocities[top])
        # we need to call transform to return to whatever format is being used
        # for the plot (?)
        self.results['intensities'].append(
            self.spectrogram.transform(intensities[top]))
        # on success we increment to the next time index
        self.time_index += 1

        if self.time_index >= len(self.time):
            return False  # we're out of data

        try:
            hop = v_high - self.results['velocities'][-2]
            hop = np.abs(hop / (velocities[1] - velocities[0]))
            return hop <= self.max_hop
        except IndexError:
            return True
        except Exception as eeps:
            print(eeps)
            return False
