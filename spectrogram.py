# coding:utf-8

"""
::
   Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
   Purpose: Compute a spectrogram from a DigFile
   Created: 9/20/19
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

from ProcessingAlgorithms.preprocess.digfile import DigFile
from UI_Elements.plotter import COLORMAPS, DEFMAP

class Spectrogram:
    """

    A Spectrogram takes a DigFile and a time range, as
    well as plenty of options, and generates a spectrogram
    using scipy.signal.spectrogram.

    Required arguments to the constructor:
        digfile: either an instance of DigFile or the filename of a .dig file

    **Optional arguments and their (default) values**

    t_start: (digfile.t0) time of the first point to use in the spectrogram
    ending:  (None) either the time of the last point or a positive integer
             representing the number of points to use; if None, the final
             point in the digfile is used.
    wavelength: (1550.0e-9) the wavelength in meters
    points_per_spectrum: (4096) the number of values used to generate
        each spectrum. Should be a power of 2.
    overlap: (7/8) the fraction of points_per_spectrum to overlap in
        successive spectra. An overlap of 0 means that each sample is used
        in only one spectrum. The default value means that successive
        spectra share 7/8 of their source samples.
    window_function: (None) the window function used by signal.spectrogram.
        The default value implies a ('tukey', 0.25) window.
    form: ('db') whether to use power values ('power'), decibels ('db'),
        or log(power) ('log') in reporting spectral intensities.
    convert_to_voltage: (True) scale the integer values stored in the
        .dig file to voltage before computing the spectrogram. If False,
        the raw integral values are used.
    detrend: ("linear") the background subtraction method.
    complex_value: (False) do you want to maintain the phase information as well.
    **Computed fields**

    time:      array of times at which the spectra are computed
    frequency: array of frequencies present in each spectrum
    velocity:  array of velocities corresponding to each spectrum
    intensity: two-dimensional array of (scaled) intensities, which
               is the spectrogram. The first index corresponds to
               frequency/velocity, the second to time.
    """

    # The following fields may be set with the spectrogram.set call using
    # kwargs
    _fields = ("points_per_spectrum",
               "overlap",
               "window_function",
               "form",
               "use_voltage",
               "detrend")

    def __init__(self,
                 digfile: DigFile,
                 t_start=None,
                 ending=None,
                 wavelength: float = 1550.0e-9,
                 points_per_spectrum: int = 4096,
                 overlap: float = 0.875,
                 window_function='hann',  # None,  # 'hanning',
                 form: str = 'db',
                 convert_to_voltage: bool = True,
                 detrend: str = "linear",
                 **kwargs
                 ):
        """
        Keyword arguments we handle:
        scaling: 'spectrum' or 'density'
        """
        if isinstance(digfile, str):
            digfile = DigFile(digfile)
        if isinstance(digfile, DigFile):
            self.data = digfile
        else:
            print(isinstance(digfile, DigFile))
            print(type(digfile))
            print(digfile)

            raise TypeError("Unknown file type")

        self.t_start = t_start if t_start != None else self.data.t0
        p_start, p_end = self.data._points(self.t_start, ending)
        self.t_end = self.t_start + self.data.dt * (p_end - p_start + 1)

        self.wavelength = wavelength
        self.points_per_spectrum = points_per_spectrum
        self.overlap = overlap
        self.window_function = window_function
        self.form = form
        self.use_voltage = convert_to_voltage
        self.detrend = detrend
        self.nfft = kwargs.get('nfft')  # handles zero padding

        # the following will be set by _calculate
        self.time = None
        self.frequency = None
        self.velocity = None
        self.intensity = None

        # deal with kwargs


        try:
            if False:
                self._load()
            else:
                raise Exception()
        except:
            self._compute(ending, **kwargs)
            # self._save()
    
    @staticmethod
    def root_code_dir():
        """
        Compute the path for the root folder that contains the code.
        """
        gitRep = "LANL_2019_Clinic" # This is the name of the root folder for the source.
        # We are operating under the assumption that the data files (dig files) will be
        # in a folder that is on the same level as this source folder.
        parent, curr = __file__, ""
        while curr != gitRep:
            parent, curr = os.path.split(parent)
        return os.path.realpath(os.path.join(parent, gitRep))

    def _compute(self, ending, **kwargs):
        """
        Compute a spectrogram. This needs work! There need to be
        lots more options that we either want to supply with
        default values or decode kwargs. But it should be a start.
        """
        if self.use_voltage:
            vals = self.data.values(self.t_start, ending)
        else:
            vals = self.data.raw_values(self.t_start, ending)

        # possible modes are 'psd', 'complex', 'magnitude',
        # 'angle', and 'phase'

        scaling = kwargs.get('scaling', 'spectrum')


        modes = ["complex", "phase", "psd"]
                
        self.availableData = modes
        self.complex = []
        self.phase = []

        for mode in modes:
            # ["complex", "magnitude", "angle", "phase", "psd"]
            # The available modes for signal.spectrogram 
            # Only need to compute complex, phase, and psd and then we can derive the rest.

            # You can probably compute the psd from the complex spectrum
            # but we do not havve the time to investigate.

            freqs, times, spec = signal.spectrogram(
                vals,
                1.0 / self.data.dt,  # the sample frequency
                window=self.window_function if self.window_function else (
                    'tukey', 0.25),
                nperseg=self.points_per_spectrum,
                noverlap=int(self.overlap * self.points_per_spectrum),
                detrend=self.detrend,  # could be constant,
                scaling=scaling,
                mode=mode,
                nfft=self.nfft
            )
            setattr(self, mode, spec)


        times += self.t_start

        self.magnitude = np.abs(getattr(self, "complex"))
        self.angle = np.angle(getattr(self, "complex"))
        self.real = np.real(getattr(self, "complex"))
        self.imaginary = np.imag(getattr(self, "complex"))

        self.availableData.extend(["magnitude", "angle", "real", "imaginary"])
        self.availableData.remove("complex")

        self.availableData.append("intensity")
        self.availableData.remove("psd")
        self.intensity = self.transform(getattr(self, "psd"))

        # Convert to a logarithmic representation and use floor to attempt
        # to suppress some noise.
        self.histogram_levels = self.histo_levels(self.intensity)

        # the first index is frequency, the second time
        self.frequency = freqs
        self.time = times

        # scale the frequency axis to velocity
        self.velocity = freqs * 0.5 * self.wavelength  # velocities

        # Now compute the probe destruction time.
        self.probeDestructionTime()
        
        self.estimatedStartTime_ = None
        # self.estimateStartTime()

    def transform(self, vals):
        """
        Perform any modification to values dictated by the value of self.form
        """
        epsilon = 1e-10
        if self.form == 'db':
            return 10 * np.log10(vals + epsilon) # Since you are already starting with power.
        if self.form == 'log':
            return np.log10(vals + epsilon)
        return vals

    def histo_levels(self, array: np.ndarray):
        """
        Sort a one-dimensional version of the array and report a vector of
        the values at 0%, 1%, ..., 100%
        """
        vals = array.flatten()
        vals.sort()
        indices = np.asarray(np.linspace(
            0, len(vals) - 1, 11), dtype=np.uint32)
        ladder = {'tens': vals[indices], }
        indices = np.linspace(indices[-2], indices[-1], 11, dtype=np.uint32)
        ladder['ones'] = vals[indices]
        indices = np.linspace(indices[-2], indices[-1], 11, dtype=np.uint32)
        ladder['tenths'] = vals[indices]
        return ladder

    def set(self, **kwargs):
        """
        Update the spectrogram to use the new parameters
        specified in the keyword arguments. If any changes cause
        the underlying values to change, recompute the spectrogram.
        """
        changed = False
        for field in self._fields:
            if field in kwargs and kwargs[field] != getattr(self, field):
                changed = True
                setattr(self, field, kwargs[field])
        if changed:
            self._compute(None)

    def __str__(self):
        return ", ".join(
            [str(x) for x in
             [self.data.filename,
              f"{self.points_per_spectrum} / {self.overlap*self.points_per_spectrum}",
              self.form,
              self.intensity.shape
              ]
             ])

    def _point_to_time(self, p: int):
        "Map a point index to a time"
        return self.time[p]

    def _time_to_index(self, t: float):
        "Map a time to a point number"
        p = (t - self.t_start) / (self.time[1] - self.time[0])
        p = int(0.5 + p)  # round to an integer
        if p < 0:
            return 0
        return min(p, len(self.time) - 1)

    def _velocity_to_index(self, v: float):
        "Map a velocity value to a point number"
        p = (v - self.velocity[0]) / (self.velocity[1] - self.velocity[0])
        p = int(0.5 + p)  # round
        if p < 0:
            return 0
        return min(p, -1 + len(self.velocity))

    def slice(self, time_range, velocity_range, complexData:bool = False, phaseData:bool = False):
        """
        Input:
            time_range: Array/Tuple/List of times (t0, t1)
                t1 should be greater than t0 but we will handle the other case
            velocity_range: Array/Tuple/List of velocities (v0, v1)
                v1 should be greater than v0 but we will handle the other case
            complexData (False):
                Do you want the complex data as well?
            phaseData (False):
                Do you want the phase data as well?
        Output:
            4 arrays time, velocity, intensity, original_spec
            time: the time values used in the measurement from t0 to t1 inclusive.
            velocity: the velocity values measured from v0 to v1 inclusive.
            intensity: the corresponding intensity values that we measured.
            original_spec: the output of the rolling FFT that we used. Depends on
                the value of self.computeMode
        """
        if time_range == None:
            time0, time1 = 0, len(self.time) - 1
        else:
            time0, time1 = [self._time_to_index(t) for t in time_range]
        if velocity_range == None:
            vel0, vel1 = 0, len(self.velocity) - 1
        else:
            vel0, vel1 = [self._velocity_to_index(v) for v in velocity_range]
        if time0 > time1:
            time0, time1 = time1, time0  # Then we will just swap them.
        if vel0 > vel1:
            # Then we will just swap them so that we can index normally.
            vel0, vel1 = vel1, vel0
        tvals = self.time[time0:time1 + 1]
        vvals = self.velocity[vel0:vel1 + 1]
        ivals = self.intensity[vel0:vel1 + 1, time0:time1 + 1]

        if complexData and phaseData:
            return tvals, vvals, ivals, self.complex[vel0:vel1+1, time0:time1+1], self.phase[vel0:vel1+1, time0:time1+1]
        elif complexData:
            return tvals, vvals, ivals, self.complex[vel0:vel1+1, time0:time1+1]
        elif phaseData:
            return tvals, vvals, ivals, self.phase[vel0:vel1+1, time0:time1+1]

        return tvals, vvals, ivals
        
    def analyze_noise(self, **kwargs):
        """
        Analyze the noise by performing a double-exponential fit.
        Possible kwargs:

            v_min (baseline)
            v_max (5000)
            t_min (0)
            t_max (probe destruction)
            n_bins (100)

        Returns:
            DoubleExponential fit object, which includes
            fields beta, lam1, lam2, amp, mean, stdev, chisq, and
            prob_greater
        """
        from fit import DoubleExponential, Exponential
        from ProcessingAlgorithms.BoundingAreaOfInterest.baselines import baselines_by_squash

        v_min = kwargs.get('v_min')
        if not v_min:
            peaks, _, hts = baselines_by_squash(self)
            v_min = peaks[0]

        v_max = kwargs.get('v_max', 5000)
        t_min = kwargs.get('t_min', self.t_start)
        t_max = kwargs.get('t_max', self.vertical_spike())
        if not t_max:
            t_max = self.t_end
        n_bins = kwargs.get('n_bins', 100)

        # extract the region and convert to power
        my_slice = self.power(self.slice((t_min, t_max), (v_min, v_max))[2])
        # set the cutoff for the maximum intensity we consider by
        # looking in histogram_levels
        cutoff = self.power(self.histogram_levels['ones'][1])

        # construct the bins
        bins = np.linspace(0, cutoff, n_bins)
        histo, edges = np.histogram(my_slice, bins=bins, density=False)
        dub = DoubleExponential(edges[2:-1], histo[2:])
        if dub.error:
            dub = Exponential(edges[2:-1], histo[2:])
        dub.v_min = v_min
        dub.v_max = v_max
        dub.t_min = t_min
        dub.t_max = t_max
        dub.n_bins = n_bins
        return dub

    def noise_level(self, percentile=0.75):
        """
        Estimate the noise level by sorting all intensities and
        returning the level of the point at the given percentile.
        """
        all_intensity = np.sort(self.intensity.flatten())
        index = int(percentile * len(all_intensity))
        return all_intensity[index]

    def squash(self, along='time', dB=False):
        """
        Sum along either rows (along = 'time') or columns
        (along = 'velocity') as power and return a one-dimensional
        array normalized to unit height. The array is
        intensity vs time if along is 'velocity' and it is
        intensity vs velocity if along is 'time'
        """
        axis = 1 if along == "time" else 0
        vals = np.sum(self.power(self.intensity), axis=axis)
        vals /= np.max(vals)
        if dB:
            vals = 10 * np.log10(vals)
        return vals

    def vertical_spike(self):
        """
        Look for an outstanding peak with intensity spread across a broad
        band of frequencies.
        """
        intensity = self.squash(along='velocity')
        peaks, props = find_peaks(
            intensity,
            height=0.1,  # squash produces a normalized output
            distance=10  # does this make any sense that spikes must
            # be at least 10 pixels apart?
        )
        if len(peaks) == 0:
            return None
        # peaks holds the time_indices
        heights = props['peak_heights']
        ordering = np.flip(np.argsort(heights))
        peak_t_indices = peaks[ordering]
        peak_times = self.time[peak_t_indices]
        peak_heights = heights[ordering]  # do I need this?
        # If we're lucky, the first peak corresponds to
        # destruction
        return peak_times[0]

    # Routines to archive the computed spectrogram and reload from disk

    def _location(self, location, create=False):
        """

        """
        if location == "":
            location = os.path.splitext(self.data.path)[0] + \
                '.spectrogram'
        if os.path.exists(location) and not os.path.isdir(location):
            raise FileExistsError
        if not os.path.exists(location) and create:
            os.mkdir(location)
        return location

    def _save(self, location=""):
        """
        Save a representation of this spectrogram.
        The format is a folder holding the three numpy arrays
        and a text file with the parameters.
        If the location is a blank string, the folder has
        the name of the digfile, with .dig replaced by .spectrogram.
        """
        location = self._location(location, True)
        with open(os.path.join(location, "properties"), 'w') as f:
            for field in self._fields:
                f.write(f"{field}\t{getattr(self,field)}\n")
        np.savez_compressed(
            os.path.join(location, "vals"),
            velocity=self.velocity,
            frequency=self.frequency,
            time=self.time,
            intensity=self.intensity)

    def _load(self, location=""):
        location = self._location(location)
        if not os.path.isdir(location):
            raise FileNotFoundError
        try:
            with open(os.path.join(
                    location, "properties"), 'r') as f:
                for line in f.readlines():
                    field, value = line.split('\t')
                    assert value == getattr(field)
            loaded = np.load(os.path.join(location, "vals"))
            for k, v in loaded.items():
                setattr(self, k, v)
            return True
        except Exception as eeps:
            print(eeps)
        return False

    @property
    def max(self):
        """The maximum intensity value"""
        return self.intensity.max()

    @property
    def min(self):
        """The minimum intensity value"""
        return self.intensity.min()

    @property
    def dv(self):
        "The velocity step size"
        return self.velocity[1] - self.velocity[0]

    @property
    def dt(self):
        "The time step size"
        return self.time[1] - self.time[0]

    @property
    def v_max(self):
        return self.wavelength * 0.25 / self.data.dt

    def power(self, values):
        """
        Given an np.array of intensity values from the spectrogram,
        return the corresponding power values (undoing any logarithms,
        if necessary).
        """
        if self.form == 'db':
            return np.power(10.0, 0.1 * values)
        if self.form == 'log':
            return np.power(10.0, values)
        return values

    def signal_to_noise(self):
        """
        Give an approximate signal to noise ratio for each time slice.
        Max/Mean
        """
        timeVelInten = np.transpose(self.intensity)
        answer = np.zeros(len(self.time))
        for ind in range(len(self.time)):
            answer[ind] = np.max(timeVelInten[ind]) / \
                np.mean(timeVelInten[ind])
        return answer

    def probeDestructionTime(self):
        """
        Compute an approximate value for the probe destruction time based
        upon the maximum total intensity for each time slice.
        """
        totalInten = np.sum(self.intensity, axis=0)
        maxInten = np.max(totalInten)
        inds = np.where(totalInten == maxInten)
        self.probe_destruction_time = self.time[inds][0]
        self.probe_destruction_index = inds[0][0]

        # Compute the maximum single intensity value and use that as another
        # estimate of probe destruction. Choose the smaller of the two options.
        a = self.max
        maxArray = np.max(self.intensity, axis=0)
        inds2 = np.where(maxArray == a)
        self.probe_destruction_index_max = inds2[0][0]
        if False:
            print("The value of PD index using the max estimate is", self.probe_destruction_index_max, "it has type", type(self.probe_destruction_index_max))
            print("The time array has shape", self.time.shape)

        self.probe_destruction_time_max = self.time[self.probe_destruction_index_max]

    def estimateStartTime(self):
        """
        Compute an approximate value for the jump off time based upon the change in 
        the baseline intensity.
        """
        from baselineTracking import baselineTracking
        from ProcessingAlgorithms.BoundingAreaOfInterest.baselines import baselines_by_squash
        peaks, _, _ = baselines_by_squash(self)
        self.estimatedStartTime_ = baselineTracking(self, peaks[0], 0.024761904761904763)

    def plotHist(self, fig = None, minFrac=0.0, maxFrac = 1.0, numBins = 1001, **kwargs):
        if fig == None:
            fig = plt.figure()
        bins = np.linspace(self.min, self.max, numBins)

        threshold = bins[int(numBins*minFrac):int(numBins*maxFrac)]

        plt.hist(self.intensity.flatten(), threshold, **kwargs)
        axes = plt.gca()
        axes.set_ylabel('Counts', fontsize = 18)
        axes.set_xlabel('Intensity', fontsize = 18)
        axes.xaxis.set_tick_params(labelsize=14)
        axes.yaxis.set_tick_params(labelsize=14)        
        title = self.data.filename.split('/')[-1]
        axes.set_title(title.replace("_", "-")+" Intensity Histogram", fontsize = 24)
        
        return fig

    def plot(self, transformData = False, **kwargs):
        # max_vel=6000, vmin=-200, vmax=100):
        pcms = {}
        if "psd" in self.availableData:
            self.availableData.append("intensity")
        if "complex" in self.availableData:
            self.availableData.append("real")
            self.availableData.append("imaginary")


        endTime = self._time_to_index((self.probe_destruction_time + self.probe_destruction_time_max)/2)
        # Our prediction for the probe destruction time. Just to make it easier to plot. 

        cmapUsed = COLORMAPS[DEFMAP]
        if 'cmap' in kwargs:
            # To use the sciviscolor colormaps that we have downloaded.
            attempt = kwargs['cmap']
            if attempt in COLORMAPS.keys():
                cmapUsed = COLORMAPS[attempt]
                del kwargs['cmap']
        top = kwargs.get("max_vel", None)
        bot = kwargs.get("min_vel", None)
        right = kwargs.get("max_time", None)
        left = kwargs.get("min_time", None)

        print("The axes settings should be t,b,r,L", top, bot, right, left)

        if top != None:
            del kwargs['max_vel']
        if bot != None:
            del kwargs['min_vel']
        if 'title' in kwargs:
            title = kwargs.pop('title')
        else:
            title = self.data.title
        if left != None:
            del kwargs["min_time"]
        if right != None:
            del kwargs["max_time"]



        for data in self.availableData:
            zData = getattr(self, data, "intensity") # getattr(object, itemname, default)
            
            key = f"{data}" + (f" transformed to {self.form}" if transformData else " raw")
            fig = plt.figure(num=key)
            axes = plt.gca()

            pcm = None # To define the scope.
            if 'cmap' not in kwargs:
                pcm = axes.pcolormesh(
                    self.time[:endTime] * 1e6,
                    self.velocity,
                    self.transform(zData[:,:endTime]) if (data != "intensity" and transformData) else zData[:,:endTime],
                    cmap = cmapUsed,
                    **kwargs)
            else:
                pcm = axes.pcolormesh(
                    self.time[:endTime] * 1e6,
                    self.velocity,
                    self.transform(zData[:,:endTime]) if (data != "intensity" and transformData) else zData[:,:endTime],
                    **kwargs)

            dataToLookAt = self.transform(zData[:,:endTime]) if (data != "intensity" and transformData) else zData[:,:endTime]

            print(f"The current maximum of the colorbar is {np.max(dataToLookAt) } for the dataset {data}")
            
            if self.estimatedStartTime_ != None:
                # Plot the start time estimate.
                axes.plot([self.estimatedStartTime_]*len(self.velocity), self.velocity, "k-", label = "Estimated Start Time", alpha = 0.75)
                # plt.legend()
            
            print(f"The current maximum of the colorbar is {np.max(zData[:,:endTime])} for the dataset {data}")
            plt.gcf().colorbar(pcm, ax=axes)
            axes.set_ylabel('Velocity (m/s)', fontsize = 14)
            axes.set_xlabel('Time ($\mu$s)', fontsize = 14)
            axes.xaxis.set_tick_params(labelsize=12)
            axes.yaxis.set_tick_params(labelsize=12)        
            title = self.data.filename.split('/')[-1]
            axes.set_title(title.replace("_", "-") + f" {data} spectrogram", fontsize = 24)

            axes.set_xlim(left, right)
            axes.set_ylim(bot, top) # The None value is the default value and does not update the axes limits.            

            pcms[key] = pcm
        
        return pcms, axes


if __name__ == '__main__':
    currDir = os.getcwd()
    sp = Spectrogram(
        '../dig/CH_4_009/seg10',
        None,
        None,
        mode=('psd', 'phase', 'angle'))
    print(sp)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(
        sp.time * 1e6,
        sp.velocity,
        sp.angle
    )
    fig.colorbar(pcm, ax=ax)
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time ($\mu$s)')
    plt.show()
    os.chdir(currDir)
