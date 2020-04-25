#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: To identify baselines in a spectrogram
  Created: 10/10/19
"""

import numpy as np
from spectrogram import Spectrogram
from scipy.signal import find_peaks
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def baselines_by_squash(
    spectrogram: Spectrogram,
    segments = 8,
    min_height = 0.0005,  # minimum fraction of the tallest peak for consideration
    min_separation = 400,  # minimum separation of peaks in m/s
    peak_width = 20,
    min_percent = 80,
    max_side_lobe = 0.5
):
    """
    Return a list of baseline velocities and their uncertainties.
    Inputs:
      -  spectrogram: an instance of Spectrogram
    Optional inputs:
      -  segments: an integer specifying the number of segments into which to
         divide the time dimension to look for a consistent peak
      -  min_height: the minimum fraction of the strongest squashed peak
         to consider when looking for baselines
      -  min_separation: the minimum separation of baselines, in m/s
      -  peak_width: to be considered a peak, look at points at this remove
         (in m/s) from a candidate peak to see that their values are lower than
         max_side_lobe times the peak's value
      -  min_percent: minimum percentage of segments satisfying the peak criterion
         described above under peak_width
      -  max_side_lobe: the segment's squashed value at ± peak_width must be
         less than or equal to max_side_lobe times the peak's value
    Outputs:
      -  peaks: an array of peaks, in descending order of strength
      -  widths: an array of uncertainty values for the peaks
      -  heights: corresponding peak heights, normalized to
            the greatest height
    """

    # Collapse along the time axis, making sure to use power,
    # not dB
    dv = spectrogram.velocity[1] - spectrogram.velocity[0]

    powers = spectrogram.power(spectrogram.intensity)

    # We'd like to squash into about 8 distinct segments
    # along axis 1 and then make sure that we get consistent
    # values before calling a peak a baseline

    boundaries = list(range(0, len(spectrogram.time),
                            len(spectrogram.time) // segments))
    boundaries.append(len(spectrogram.time))
    squash = []
    for n in range(segments):
        squash.append(
            powers[:, boundaries[n]:boundaries[n + 1]].mean(axis=1))

    combined_spectrum = spectrogram.power(
        spectrogram.intensity).mean(axis=1)
    tallest = combined_spectrum.max()
    # this should be improved
    peaks, properties = find_peaks(
        combined_spectrum,
        height=min_height * tallest,  # peaks must be this tall to count
        distance=min_separation / dv  # peaks must be separated by this much at minimum
    )
    heights = properties['peak_heights']
    peak_ht = heights.max()
    # produce an ordering from tallest to shortest peak
    ordering = np.flip(np.argsort(heights))
    peak_positions = peaks[ordering]
    peaks = spectrogram.velocity[peak_positions]
    heights = heights[ordering] / peak_ht

    # Now we want to filter out the peaks that don't correspond to
    # consistent signal across the segments

    strong_peaks = []
    delta = int(peak_width / dv)
    for n, pk_pos in enumerate(peak_positions):
        if pk_pos >= delta:
            totes = 0
            for sq in squash:
                try:
                    if sq[pk_pos + delta] < max_side_lobe * sq[pk_pos] > sq[pk_pos - delta]:
                        totes += 1
                except IndexError:
                    pass
            strong_peaks.append(totes >= segments * min_percent * 0.01)
            # print(f"[{pk_pos}] -> {spectrogram.velocity[pk_pos]} ({heights[n]})) got {totes}")
        else:
            strong_peaks.append(False)
    keepers = np.array(strong_peaks)

    return peaks[keepers], (np.ones(len(peaks)) * dv)[keepers], heights[keepers]


def baselines_by_fft(spectrogram):
    """
    Return a list of baseline velocities and their uncertainties.

    Inputs:
      -  spectrogram: an instance of Spectrogram

    Outputs:
      -  peaks: an array of peaks, in descending order of strength
      -  widths: an array of uncertainty values for the peaks
      -  heights: corresponding peak heights, normalized to
            the greatest height

    Observations:
        This routine probably does a poor job of discriminating against
        lone peaks arising from an anomalously large value at a fairly
        isolated time. It might be better to chop the range into several
        intervals and insist on a significant peak in each.
    """

    assert isinstance(spectrogram, Spectrogram)
    # Perform a single Fourier transform of the voltage data in the range
    # corresponding to this spectrogram, rounded down to the nearest power
    # of 2 (we could also zero pad up to the next higher power of two).
    first_point, last_point = spectrogram.data._points(None, None)
    nPoints = last_point - first_point + 1
    octaves = np.log2(nPoints)  # how many octaves are spanned
    num_points = int(2 ** np.floor(octaves))
    vals = spectrogram.data.values(None, num_points)
    # Now compute the power spectrum of these points
    # At present, we aren't worrying about a window function, assuming that
    # the noise spread across all frequencies from abrupt shifts will not
    # worry the peaks significantly.
    cspec = fft(vals)[0:1 + num_points // 2]
    powers = np.power(np.abs(cspec), 2)

    velocities = np.linspace(0.0, 0.25 / spectrogram.data.dt,
                             len(cspec)) * spectrogram.wavelength
    # At this point, powers vs velocities should show some very
    # narrow spikes. Let's normalize the array of powers
    peak_power = powers.max()
    powers /= peak_power

    peaks, properties = find_peaks(
        powers,
        height=0.001,   # peaks must be this tall to count
        # peaks must be separated by this much at minimum
        distance=len(powers) // 1024,
    )
    heights = properties['peak_heights']
    ordering = np.flip(np.argsort(heights))

    # sort the peaks and heights into descending order
    peaks, heights = peaks[ordering], heights[ordering]

    neighborhoods = []
    width = 50
    for peak_center in peaks:
        low, high = max(0, peak_center -
                        width), min(len(powers) - 1, peak_center + width)
        neighborhoods.append([velocities[low:high], powers[low:high]])
    return neighborhoods


if __name__ == '__main__':
    import os
    os.chdir('../dig')
    sgram = Spectrogram('./CH_4_009/seg00.dig', 0.0, 50.0e-6)


    peaks, uncertainties, peak_heights = baselines_by_squash(sgram)
    velos = []
    times = [x for x in range(0,30)]


    pcms, axes = sgram.plot(min_time=0, min_vel=100, max_vel=10000, cmap='3w_gby')


    # add the peaks to the plot and change the color map range
    pcm = pcms['intensity raw']
    pcm.set_clim(-40, -65)
    for peak in peaks:
        velo_index = sgram._velocity_to_index(peak)
        axes.plot(times, [peak for x in range(0,30)], color='red', linewidth=3, markersize=15)

    plt.show()
