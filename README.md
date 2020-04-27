# LANL 2019-2020 Clinic

This project is the product of the Los Alamos National Lab 2019-2020 Harvey Mudd Clinic Team

## Purpose

The purpose of this project is to analyze PDV data, and generate useful information about the traces contained in them. This repository contains our tools for extracting the voltage data from those files, converting it to useful velocity data, and analyzing that data to obtain a trace of the surface velocity over an experiment.

## Getting Started

### Python 3.6+ Required

### Required packages

Use the package manager (either pip or conda is compatible) to install required packages. From the root directory of the project run:


```bash
pip install -r requirements.txt
or
conda install --file requirements.txt
```
 
### File Structure
This repository is designed with assumption of the structure of files. The repository as stored here is correct in managing locations, however in a local clone some hard coded locations may need to be updated.

In general we expect a layout of

```
Project_root
│       
│
└───LANL_2019_Clinic // Contains the analysis files
│   │   README.md
│   │   spectrogram.py
│   │   ...
│   │
│   └───ProcessingAlgorithms
│   │   │   spectrum.py
|   |   |   ...
│   └───ImageProcessing
|       | Template Matching Files
│       │   ...
└───dig // Contains the data
│   │   digFile1.dig
│   │   digFile2.dig 
│   │
│   └───digfolder1
│       │   experiment1.dig
│       │   experiment2.dig
│       │   ... 
```
* **Note:** As shown above, the dig *folder* is not in the git tracked section of the repository. It is a folder at the same level as the git root. This is because the data is not used with git nor github.

 ## Usage
 The project can be used in a variety of ways, however the intention is to use the `Spectrogram` class. Here are two simple ways to generate a spectrogram from a .dig file.

 ### `Spectrogram` class
 From a file or the python interpreter, this is invoked by creating an instance of the class.

 ```python
 spec = Spectrogram("PDV.dig")
 ```
In order to then display the spectrogram of this file, the following command is used:

```python
spec.plot()
```
This will print the entire .dig file represented as a spectrogram. If desired, `max_vel=INT, min_vel=INT` can be inputed as arguments to `plot` to quickly cut the velocity axis to a specified range.

Many more options and optional arguments are available, as described in the documentation for the spectrogram class.


### Pipeline file
From `pipeline.py` batches of files can be processed at the same time, and the traces extracted. To do this:

1) Create a subdirectory with only the `.dig` files to be extracted.
2) Create a JSON file with the names and start points of the files.
3) Execute 
```bash
python pipeline.py --json_name=NAME_OF_JSON_FILE
```

The output of this program will be stored in a direcotry. More information can be found in the documentation

## Test Data
Unfortunately, public test data is not available for this project. This is due to the procedures involving the .dig files at Los Alamos. Additionally, the files are often over a gigabyte in size, holding millions of data points, which makes them ungainly to store remotely.



## Authors

* **Nick Koskelo** - *Project Manager*
* **Max Treutelaar**
* **Trevor Walker**
* **Rikki Walters** - *Spring Member*
* **Isabel Duan** - *Spring Member*
* **Professor Peter Saeta** -*Faculty Advisor*