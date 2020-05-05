#!/usr/bin/env python3
# coding:utf-8

"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Process a set of .dig files
  Created: 02/21/20

  This file should reside at the base level in the
  LANL_2019_Clinic directory, and should be made executable
  for greatest ease of use.

  To run it, create a file called script.txt in a fresh directory
  (which I'll call $HOME) and populate it with some of the available
  commands defined in the present file. An example script:

  make_spectrogram overlap=0.5 points_per_spectrum=32768
  find_baselines
  find_signal t_start = 2e-5
  follow_signal
  gaussian_fit

  You can generate a list of available commands by calling

  python3 pnspipe.py --help

  The general structure of the script file is the name of the command
  to run followed by any keyword=value pairs you wish to pass to the
  command.

  The commands in the script.txt file will be run on each of the files
  in the dig folder that are consistent with the --regex argument (which
  defaults to '.*' to operate on all files) and are not excluded by the
  optional --exclude argument. For each .dig file that is consistent,
  a subfolder of $HOME is created in a parallel hierarchy. That folder
  contains a log file reporting the stages of analysis and any
  output files generated by the various operations.
"""

import datetime
import os
import re
from time import time
from ProcessingAlgorithms.preprocess.digfile import DigFile
import concurrent.futures


class PNSPipe:
    """
    Pipeline to process a single .dig file by executing a number of operations,
    as described in a sequence called 'orders'. Any parameters that need to be
    supplied may be specified in keyword arguments; individual operations can
    scan the keyword arguments for any parameters they require.
    """

    def __init__(self, filename, orders, **kwargs):
        """
        On entry we assume that we are in the directory where
        the report and output should be written, in a folder
        hierarchy that mirrors the structure to the .dig file.
        """
        self.home = os.getcwd()
        self.filename = filename
        self.orders = orders
        self.df = DigFile(filename)
        self.rel_dir = os.path.join(self.df.rel_dir, self.df.basename)
        self.output_dir = os.path.join(self.home, self.rel_dir)
        # Set a title, which can be used in filenames to describe this segment
        self.title = self.df.title.replace("/", "-")

        # Make sure that this directory exists
        os.makedirs(self.rel_dir, exist_ok=True)
        print(f"Making directories {self.rel_dir}")
        # Open a log file
        self.logfile = open(
            os.path.join(self.output_dir, 'info.txt'),
            'w', buffering=1)

        now = datetime.datetime.now()
        self.log(
            f"{self.filename} log, {now.strftime('%a %d %b %Y at %H:%M:%S')}")
        self.spectrogram = None
        self.baselines = []
        self.gaps = []
        self.jumpoff = None
        self.probe_destruction = None
        self.followers = []
        self.signals = []
        self.pandas_format = dict(
            time=lambda x: f"{x*1e6:.3f}",
            peak_v=self.onedigit,
            peak_int=self.twog,
            g_center=self.onedigit,
            g_width=self.onedigit,
            g_bgnd=self.twog,
            g_int=self.twog,
            g_chisq=self.onedigit,
            g_prob=self.twog,
            g_scale=self.onedigit,
            m_center=self.onedigit,
            m_width=self.onedigit,
            m_bgnd=self.twog,
            amplitude=self.onedigit,
            dcenter=self.twodigit,
        )

        # Routines can store results in the results dictionary.
        # We keep a catalog of the results dictionaries for further
        # processing post.
        self.results = dict()
        # How do we specify spectrogram parameters?
        # If the first command is not a specification for computing
        # a spectrogram, use the defaults

        from Pipeline.persegment import make_spectrogram
        if orders[0][0] != make_spectrogram:
            orders.insert(0, (make_spectrogram, {}))

        for order in orders:
            routine, kwargs = order
            self.start(routine, kwargs)
            print(f"{routine.__name__} ...", end="", flush=True)
            routine(self, **kwargs)
            print(self.end())

    @property
    def segment_parent(self):
        """
        Return the path to the segment's parent folder, if we are processing
        a segment, otherwise return the output_dir.
        """
        if self.df.is_segment:
            return os.path.split(self.output_dir)[0]
        return self.output_dir

    @property
    def segment_name(self):
        """
        Return a properly neutered name identifying the segment
        """
        return self.df.title.replace("/", "-")

    def onedigit(self, val):
        return f"{val:.1f}"

    def twodigit(self, val):
        return f"{val:.2f}"

    def twog(self, val):
        return f"{val:.2g}"

    def log(self, x, echo=False):
        print(x, file=self.logfile)
        if echo:
            print(x)

    def open(self, message: str, kwargs: dict):
        """
        Start a frame with label message. If kwargs is not
        empty, include the parameters in the dictionary.
        """
        self.log(f"<<<<< {message}")
        if kwargs:
            for k, v in kwargs.items():
                self.log(f"+ {k} = {v}")

    def close(self, message: str):
        msg = f">>>>> {message}"
        self.log(msg + "\n\n")
        return msg

    def start(self, routine, kwargs):
        """
        Call this on entry to get the caller's information
        added to the log and to start the timer for the
        routine.
        """
        self.t0 = time()
        self.open(routine.__name__, kwargs)

    def end(self):
        """
        Call this on exit
        """
        return self.close(self._timestr(time() - self.t0))

    def _timestr(self, dt):
        if dt > 0.1:
            unit = 's'
        elif dt > 0.001:
            dt *= 1000
            unit = 'ms'
        elif dt > 1e-6:
            dt *= 1e6
            unit = 'µs'
        return f"{dt:.2f} {unit}"

    def __del__(self):
        try:
            # print("I'm in the destructor")
            self.logfile.flush()
            self.logfile.close()
            os.chdir(self.home)
            print(os.getcwd())
        except:
            pass


# def gaussian_fit(pipe: PNSPipe, **kwargs):
    # """
    # oops
    # """
    # for signal in pipe.signals:
        # First add the requisite columns
        # blanks = np.zeros(len(signal)) + np.nan
        # signal['center'] = blanks
        # signal['width'] = blanks
        # signal['amplitude'] = blanks
        # signal['dcenter'] = blanks
        # signal['mean'] = blanks
        # signal['sigma'] = blanks

        # for n in range(len(signal)):
            # row = signal.iloc[n]
            # t_index = row['t_index']
            # vpeak = sg._velocity_to_index(row['velocity'])
            # vfrom, vto = row['vi_span']
            # vfrom, vto = vpeak - neighborhood, vpeak + neighborhood
            # powers = sg.intensity[vfrom:vto, t_index]
            # speeds = sg.velocity[vfrom:vto]
            # mom = moment(speeds, powers)
            # signal.loc[n, 'mean'] = mom['center']
            # signal.loc[n, 'sigma'] = mom['std_dev']
            # gus = Gaussian(
            # speeds, powers,
            # center=row['velocity'],
            # width=sg.velocity[2] - sg.velocity[0]
            # )
            # if gus.valid:
            # signal.loc[n, 'center'] = gus.center
            # signal.loc[n, 'width'] = gus.width
            # signal.loc[n, 'amplitude'] = gus.amplitude
            # diff = gus.center - signal.loc[n, 'velocity']
            # signal.loc[n, 'dcenter'] = diff
            # discrepancy = abs(diff / gus.width)
            # if discrepancy > sigmas:
            # The difference between the peak follower and the gaussian
            # fit was more than 1 value of the width.
            # Let's print out a plot to show what's going on
            # plt.clf()
            # vmin, vmax = sg.velocity[vfrom], sg.velocity[vto]
            # make sure we have all the freshest values
            # row = signal.iloc[n]

            # plt.plot([row['velocity'], ],
            # [row['intensity'], ], 'k*')
            # plt.plot([row['mean'] + n * row['sigma'] for n in (-1, 0, 1)],
            # [row['intensity'] for x in (-1, 0, 1)],
            # 'gs')
            # plt.plot(speeds, powers, 'r.')
            # vels = np.linspace(gus.center - 6 * gus.width,
            # gus.center + 6 * gus.width, 100)
            # plt.plot(vels, gus(vels), 'b-', alpha=0.5)
            # tval = f"${pipe.spectrogram.time[t_index]*1e6:.2f}"
            # plt.title(tval + "$~ µs")
            # plt.xlabel("Velocity (m/s)")
            # plt.ylabel(r"Intensity")
            # plt.xlim(vmin, vmax)
            # plt.savefig(os.path.join(pipe.output_dir, f'bad{t_index}.pdf'))
            # plt.close()
            # with open(os.path.join(pipe.output_dir, f'bad{t_index}.txt'), 'w') as f:
            # f.write(pd.DataFrame(
            # {'power': powers, }, index=speeds).to_csv())

    # for signal in pipe.signals:
        # pipe.log(signal.to_string(
            # formatters=pipe.pandas_format, sparsify=False))
        # pipe.log("\n\n")
        # plt.clf()
        # fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
        # top, middle, bottom = axes
        # top.semilogy(signal.time * 1e6, signal.intensity)
        # top.set_ylabel('Intensity')
        # top.set_title(pipe.df.basename)
        # middle.plot(signal.time * 1e6, signal.velocity)
        # middle.set_ylabel(r'$v~(\mathrm{m/s})$')
        # bottom.errorbar(signal['time'] * 1e6,
            # signal.dcenter, yerr=signal.width, fmt='b.',
            # markersize=1.0, lw=0.5)
        # bottom.set_xlabel(r'$t~(\mu \mathrm{s})$')
        # bottom.set_ylabel(r'$\delta v~(\mathrm{m/s})$')
        # plt.savefig(os.path.join(pipe.output_dir, 'gauss.pdf'))


def decode_arg(x):
    """
    Attempt to convert the value x to an int, a float, or a string
    """
    x = x.strip()
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x


def process_command_file(x):
    """
    Read a text file of commands that has the form
    command1 keyword1=arg1 keyword2=arg2
    command2 keyword3=arg3
    ...

    Returns:

        A list of 2-tuples in which the first is the function
        (not the function name) to call and the second
        is a dictionary of keyword arguments.
    """
    from Pipeline import _pipe_functions
    order_text = open(x, 'r').readlines()
    orders = {x: [] for x in _pipe_functions.keys()}

    def find_function(s):
        for k, v in _pipe_functions.items():
            hit = [x for x in v if x.__name__ == s]
            if hit:
                return (k, hit[0])
        return (None, None)

    for line in order_text:
        # first remove any white space around equal signs
        line = re.sub(r' *= *', "=", line)
        fields = line.strip().split()
        if not fields:
            continue
        routine = fields.pop(0).strip(',')
        if routine:
            # func = globals()[routine]
            category, func = find_function(routine)
            kwargs = {}
            for x in fields:
                k, v = x.split('=')
                kwargs[k] = decode_arg(v)
            orders[category].append((func, kwargs))
    return orders


# Register all pipe functions in the present file

class PNSPipeline:
    """
    This class runs a pipeline process, calling PNSPipe on each
    of the requested source .dig files/segments.
    """

    def __init__(self, *args):

        from re import compile
        self.start_time = time()
        self.start_dir = os.getcwd()
        self.results = dict()
        self.finished_segments = dict()

        parser = self.make_parser()
        self.args = parser.parse_args(*args)

        self.find_files()
        if self.args.delete:
            self.delete_files()  # remove any results from a previous run
        self.include = compile(self.args.regex)
        self.exclude = compile(
            self.args.exclude) if self.args.exclude else None

        if self.args.dry:
            print(f"Dry run, storing output in base directory {os.getcwd()}")
        self.threads = self.args.threads

    def make_parser(self):
        from Pipeline import describe_pipe_functions
        import argparse

        epilog = "Available operations:\n\n" + describe_pipe_functions("")
        parser = argparse.ArgumentParser(
            description='Run a pipeline to process .dig files',
            prog="pipe",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=epilog
        )
        parser.add_argument('-q', '--quiet', help="Don't report progress")
        parser.add_argument('-s', '--segments', action='store_false',
                            help="Only process segments")
        parser.add_argument('-r', '--regex', default=r'.*',
                            help="Regular expression to select files; defaults to '.*'")
        parser.add_argument('-e', '--exclude', default=None,
                            help="Regular expression for files to exclude")
        parser.add_argument('-i', '--input', default=None,
                            help="Input file of commands; defaults to script.txt")
        parser.add_argument('-o', '--output', default=os.getcwd(),
                            help="top directory for results")
        parser.add_argument('-d', '--delete', action='store_true',
                            help="Delete existing files before the run")
        parser.add_argument('--dry', action='store_true',
                            help='List the files that would be handled and where the output would be written')
        parser.add_argument('-t', '--threads', default=1,
                            help='Run this many threads in parallel')
        return parser

    def find_files(self):
        """
        Look for the input script file
        """
        infile = ""
        args = self.args
        if args.input:
            if os.path.isfile(args.input):
                infile = os.path.abspath(args.input)
        else:
            infile = 'script.txt'

        if args.output:
            os.chdir(args.output)

        # Look for marching orders in a file called script.txt
        assert os.path.isfile(
            infile), f"You must supply file script.txt in the output directory"
        self.orders = process_command_file(infile)

    def delete_files(self):
        # remove all contents of subdirectories of the output directory first
        for candidate in os.listdir('.'):
            if os.path.isdir(candidate):
                shutil.rmtree(candidate, ignore_errors=True)

    def preprocess(self):
        for func, kwargs in self.orders['preprocess']:
            func(self.include, self.exclude, **kwargs)

    def postsegment(self, source: str):
        """
        All segments associated with the source file have been
        processed. Run all postsegment routines on this file.
        The source is the rel_dir of the source file.
        """
        # Fetch the results for all the segments of this source
        # and run any postsegment routines on the list
        # The source string is the rel_dir of the parent
        _, src = os.path.split(source)[1]
        # We now want to extract all results corresponding to this file
        pat = re.compile(src + r'/seg(\d\d)')
        segs = sorted([x for x in self.results.keys() if pat.search(x)])
        the_segments = [self.results[x] for x in segs]
        for routine, kwargs in self.orders['postsegment']:
            routine(source, the_segments, **kwargs)

    def add_segment_result(self, pipe):
        df = pipe.df
        title = df.title

        self.results[title] = pipe.results
        # Now see if all segments have completed and we can run the
        # postsegment routines
        if df.is_segment:
            src = df.source
            self.num_segments[src] -= 1
            if self.num_segments[src] == 0:
                self.postsegment(df.rel_dir)

    def segments(self):
        paths = []
        segs = {}
        dd = DigFile.dig_dir()
        for file in DigFile.all_dig_files():
            if not self.include.search(file):
                continue
            if self.exclude and self.exclude.search(file):
                continue
            path = os.path.join(dd, file)
            df = DigFile(path)
            if self.args.segments:
                if not df.is_segment:
                    continue
            paths.append(path)
            src = df.source
            if src not in segs:
                segs[src] = 1
            else:
                segs[src] += 1

        self.paths = paths
        self.num_segments = segs

        if not paths:
            print("No files were selected")
            return

        if self.threads > 1:
            print("""
            ***** WARNING *****
            Matplotlib is not thread-safe. If your onsegment routines
            include calls to matplotlib, Python will crash. So be
            forewarned!
            ***** ACHTUNG *****
            """)

            def run_pipe(path, dry, orders):
                if dry:
                    df = DigFile(path)
                    return df.title
                return PNSPipe(path, orders['onsegment'])

            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                future_runs = {executor.submit(
                    run_pipe, path, self.args.dry, self.orders): path for
                    path in paths}

                for run in concurrent.futures.as_completed(future_runs):
                    path = future_runs[run]
                    try:
                        pipe = run.result()
                        self.add_segment_result(run.result())

                    except Exception as eeps:
                        print('%r generated an exception: %s' % (path, eeps))
        else:
            for path in paths:
                if self.args.dry:
                    print(path)
                    df = DigFile(path)
                    self.results[df.title] = "Sure!"
                else:
                    try:
                        pipe = PNSPipe(path, self.orders['onsegment'])
                        self.add_segment_result(pipe)

                    except Exception as eeps:
                        print(f"\n********************\n")
                        print(f"Encountered error {eeps}")
                        print(f"while processing  {path}")
                        print("\n**********************\n")

    def postprocess(self):
        """
        Run any requested procedures on the entirety of the results
        """
        # First organize the results by base file
        # The keys are the df.title fields, which may have one or more
        # slashes. If we are doing segments, we want the penultimate 2.
        # That's all we really should be doing, so let's go with that.
        # Prepare the sources dictionary whose keys are the files
        # and whose values are dictionaries with key
        # sources = [file][segment][results]

        sources = dict()
        for key in self.results.keys():
            fields = key.split('/')
            source, segment = fields[-2:]
            if source not in sources:
                sources[source] = dict()
            sources[source][segment] = self.results[key]

        # Now run the requested commands
        orders = self.orders['postprocess']
        if orders:
            for fname, source in sources.items():
                segments = sorted(list(source.keys()))
                for order in orders:
                    routine, kwargs = order
                    routine(fname, source, segments, **kwargs)

    def run(self):
        self.preprocess()
        self.segments()
        self.postprocess()

        # restore the working directory (is this necessary?)
        os.chdir(self.start_dir)
        dt = time() - self.start_time
        print(f"Analysis took {dt:.1f} s")


if __name__ == '__main__':

    pipeline = PNSPipeline()
    pipeline.run()


