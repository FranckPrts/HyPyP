#!/usr/bin/env python
# coding=utf-8

"""
Input/Output (io)

| Option | Description |
| ------ | ----------- |
| title           | io.py |
| authors         | Franck Porteous, Jonas Mago, Guillaume Dumas |
| date            | 2023-02-01 |
"""

import mne
import numpy as np
import os
import pyxdf
import warnings

class XDF_IMPORT():
    """
    Read an XDF file and enable to export stream in convinient format (e.g., an EEG stream into a mne.Raw instance).

    Arguments:
      path: Path to LSL data (i.e., XDF file). Can be absolute or relative.
      type: Define which type of stream the user is looking to convert.
      stream_idx: List of the stream index(es) in the XDF the user whish to convert (can be `str` which the class will try to match to the name of an existing stream or an `int` which will be interpreted as such).
      sfreq: Sampling frequency, can either be set given by the user or automatically by the class.
      print_stream_name: Wheather the class should print the streams and their respective index in the XDF.
      convert_all_eeg: Boolean indicating if the class must automaticaly convert all EEG stream(s) it finds (when set to True).
      eeg_montage: A path to a local Dig montage or a mne standard montage 
    """

    def __init__(self, path: str, type: str='EEG', stream_idx: list=None, sfreq: int=None, print_stream_name: bool=True, convert_all_eeg: bool=False, save_FIF_path:bool=None, eeg_montage: str=None):
        
        self.path = path 
        self.sfreq = sfreq
        self.data, self.header = pyxdf.load_xdf(path, verbose=None)
        self.save_FIF_path = save_FIF_path
        self.eeg_montage = eeg_montage

        if print_stream_name==True: 
            self.print_stream_names()
        
        # The usr wants all possible eeg stream to be converted
        if convert_all_eeg:
            self.stream_idx = self.find_id(type)  # Find the idx for all available eeg stream(s)
            self.convert_stream_to_MNE()          # Convert them to mne.Raw
            self.montage_setup(eeg_montage)       # Set the given montage (local path or mne default montage)
        
        # The usr doesn't want automatic convertion and have not provided stream(s) idx to convert
        elif convert_all_eeg != True and stream_idx==None:  
            print("\nRun XDF_IMPORT again with convert_all_eeg=True or specify the stream(s) you wish to convert using the list arg stream_idx.")
            print("To see the available streams in your XDF, re-run XDF_IMPORT with print_stream_name=True.")
        
        # The usr has given the (specific) idx for all eeg stream(s) they wish to convert
        elif convert_all_eeg != True and stream_idx!=None:
            self.keyword_to_idx(stream_idx)   # Call keyword_to_idx to find EEG stream's idx w/ partial names 
            self.convert_stream_to_MNE()                        # Convert them to mne.Raw 
            self.montage_setup(eeg_montage)
        else:
            self.warn_and_break("please check your input") 
    
    def get_streams_names (self):
        """
        Saves the names of the XDF streams in a list (`self.channel_names`).

        Note:
            Each stream's number is associated with the ID within the XDF, not its index.
          
        """
        self.channel_names = []
        for i in range (len(self.data)):
            name = self.data[i]['info']['name'][0]
            self.channel_names.append(name)

    def keyword_to_idx(self, idx: list):
        """
        Interpret the query made by the user (a list of indexes, or `str` that matches 
        streams' name) into a list containing the indexes within the XDF file.

        Arguments:
            idx: List containing the index that the user is trying to convert.
        """
        self.get_streams_names()
        self.stream_idx = []
        for keyword_idx,keyword in enumerate(idx):
            if type(keyword)==int: # if usr gives idx
                self.stream_idx.append(keyword)
            else: # If usr gives anything other than the idx

                list_lower = [x.lower() for x in self.channel_names]
                keyword_lower = keyword.lower()
                tmp_list = [i for i, elem in enumerate(list_lower) if keyword_lower in elem] # Match keywords to string
                if len(tmp_list) == 1:
                    self.stream_idx.append(tmp_list[0]) 
                else:
                    print ("There are %i options that match your query" % len(tmp_list))
                    print ('matching streams are:')
                    for option in tmp_list:
                        print("\t", self.data[option]['info']['name'][0])
                    self.warn_and_break('Not converting any stream, please be more specific the name(s) stream(s) you wish to convert or use the argument convert_all=True')

    def convert_stream_to_MNE(self):
        """
        A function that centralizes the pipeline for creating a dictionary containing converted
        XDF stream into `mne.Raw`.

        Note:
            The returned dictionary has the name of the stream as a key and the `mne.Raw` object as the value.
        """
        self.raw_all={}
        print("\nConverting EEG stream(s) ... ")
        
        # Find if all the stream have unique names (true if any stream name is duplicated)
        eeg_stream_names = [self.data[i]['info']['name'][0] for i in self.stream_idx]
        self.duplicated_name = len(set(eeg_stream_names)) != len(eeg_stream_names) # a bool
        if self.duplicated_name:
            print("---! Carefull!\n---! Multiple stream have the same name\n---! Adding original streams' index as suffixes to the generated raws") 
        
        for i in self.stream_idx:
            print('\n---> Converting {}'.format(self.data[i]['info']['name'][0]))
            self.get_sampling_freq(i)
            self.create_info(i, type="eeg")
            self.create_raw(i, self.info, bounds=None)
            
            # Check whether the Raw's name is already used, add an incremental suffix if needed
            if self.duplicated_name: stream_name = '{}-StreamIndex-{}'.format(self.data[i]['info']['name'][0], i)
            else: stream_name = self.data[i]['info']['name'][0]
            
            # Save the object/stream_name pair in raw_all
            self.raw_all[stream_name] = self.raw

            # Save file is asked too
            if self.save_FIF_path is not None: 
                os.makedirs(self.save_FIF_path, exist_ok=True)
                self.raw.save(f"{self.save_FIF_path}{stream_name}.fif", overwrite=True)
                print(f'-> saved {stream_name} at {self.save_FIF_path}')

        print("\nConvertion done.")
        
    def find_id(self, type: str = "EEG") -> list:
        """
        Read the XDF file to find & store the XDF stream's indexes that match the `type` (e.g., "EEG"). 

        Arguments:
          type: The string (e.g., "EEG", "video") that will be matched to XDF stream's `type` to find their indexes.

        Returns:
          eeg_streams_idx: A list containing the indexes where the specified type of streams can be found.

        Note:
          By default, the stream type to find id(s) for is set to "EEG".
        """

        # Look in self.data for the ID of the streams conmtaining EEG data
        ID_eeg = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.path),[{"type": type.upper()}])

        ######################### TO DELETE
        # # Look at the name of the available streams and find those how says 'EEG'
        # for stream in range (len(self.data)):
        #     if 'EEG' in self.data[stream]['info']['name'][0]:
        #         print('{}'.format(self.data[stream]['info']['name']))

        # Define an empty list to store the indexes of the EEG stream `in streams`
        eeg_streams_idx = list()

        print("\nLooking for {} stream(s)".format(type))
        
        # For all identified EEG streams, find and store the XDF stream's idx in eeg_streams.
        for eeg_stream_id in ID_eeg: 
            for stream_idx in range(len(self.data)):
                if self.data[stream_idx]["info"]["stream_id"] == eeg_stream_id:

                    # store last EEG stream object for the assertion
                    eeg_stream = self.data[stream_idx]
                    
                    print('\tFound {} stream {} at index: {}'.format(
                        type, 
                        self.data[stream_idx]['info']['name'][0],
                        stream_idx))

                    # Store the index of the EEG streams in EEG_streams_idx
                    eeg_streams_idx.append(stream_idx)
                    break

        # Assert that we have found at least one real EEG stream
        if len(eeg_stream) != 0:
            print("\n\t--> Found {} {} stream(s) at index {}".format(len(eeg_streams_idx), type, eeg_streams_idx))
        else:
            warnings.warn('No EEG stream(s) were found in this XDF file')

        return eeg_streams_idx

    def print_stream_names (self):
        """
        Print a list of available streams and their corresponding index.

        This function iterates through the available streams and prints the stream name along with its index. 
        It also stores the stream names in the `channel_names` attribute of the object.
        
        Note:
            The number associated with each stream is its ID within the XDF; not its index.
        """

        self.channel_names = []
        print("List of available stream(s):")
        for i in range (len(self.data)):
            name = self.data[i]['info']['name']
            self.channel_names.append(name)
            print ("\tStream {} is at idx:{}".format(name[0], i))

    def get_sampling_freq (self, idx: int):
        """
        Gather sampling rate information from the XDF's EEG stream metadata.

        Arguments:
            idx: The index of the stream to get the sampling rate from.
        """
        self.sfreq = float (self.data[idx]["info"]["nominal_srate"][0])
        print (f"sampling freq is {self.sfreq} Hz")

    def auto_scale_data(self, data, std_threshold=1e-5, amplitude_threshold=1.0):
        """
        Automatically scale EEG data to V if necessary based on both standard deviation and amplitude range.

        Arguments:
            data: EEG data array.
            std_threshold: Threshold for standard deviation to determine scaling.
                           Default is set to 1e-5.
            amplitude_threshold: Threshold for amplitude range to determine scaling.
                                 Default is set to 1.0.

        Returns:
            Scaled EEG data array.
        """
        # Check for NaN and Inf values in the data

        data = data.astype(np.float64)

        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("Data contains NaN or Inf values >> NOT converting to V")
            return data

        std = data.std()
        self.dev = data

        amplitude_range = data.max() - data.min()

        if std > std_threshold and amplitude_range > amplitude_threshold:
            # Data has both a small standard deviation and small amplitude range, indicating it's in mV
            print(f"Std ({std}) and amplitude range ({amplitude_range}) are larges >> converting to V (e.g., * 10e-6)")
            scaled_data = data * 10e-6  # Scale to V
        else:
            # Data has a large standard deviation or amplitude range, indicating it's already in V
            print(f"Std ({std}) or amplitude range ({amplitude_range}) are smalls >> NOT converting to V")
            scaled_data = data
        
        return scaled_data

    def create_info (self, idx: int, type: str = "eeg"):
        """
        Create a mne.info object from the XDF's EEG stream metadata.

        Arguments:
            idx: The index of the stream to create mne.info for.
            type: Type of the stream to create info for (e.g., "EEG", "MEG").
        """
        
        #gather metadata
        n_channels = int(self.data[idx]['info']['channel_count'][0])
        ch_names = [f'EEG_{n:03}' for n in range(1, 1+n_channels)] # Use the ch names (if available in XDF) // much more to levrage from eeg.data[1]['info']...
        ch_types = [type]*n_channels

        # Create mne.Info object
        self.info = mne.create_info(ch_names, ch_types=ch_types, sfreq=self.sfreq, )
        
        # Create and add subject_info to self.info
        if self.duplicated_name:
            self.info["subject_info"] = {"his_idstr":'{}-StreamIndex-{}'.format(self.data[idx]['info']['name'][0], idx)}
        else:
            self.info["subject_info"] = {"his_idstr":self.data[idx]['info']['name'][0]}

    def create_raw (self, idx: int, info, bounds: list = None):
        """
        Create a mne.Raw object.

        Arguments:
            idx: The index of the stream to create info for.
            info: The `mne.info` to create the mne.Raw with.
            bounds: Controle the boundaries of the segment of data to convert to mne.Raw.

        Note:
            The `bounds`argument can be used to cut a specific piece of the time series:  
                - it must be a list of two values,
                - the first value is the first time point, 
                - the second value is the last time point to include in the sample
        """

        # Here we check wether the data is in the correct shape () and transpose it if necessary
        # ! We assume that no EEG recoding would have more channels than sample point !
        if self.data[idx]["time_series"].shape[0] > self.data[idx]["time_series"].shape[1]:
            data = self.data[idx]["time_series"].T
        else: 
            data = self.data[idx]["time_series"]

        # Apply automatic scaling to the data
        scaled_data = self.auto_scale_data(data)
        
        # Here we cut the data to convert if the user has given bounds
        if bounds != None:
            data = scaled_data [:,bounds[0]:bounds[1]]
        
        # Create the mne.Raw
        self.raw = mne.io.RawArray(scaled_data, info) 
        
        # Rename channels if the information is available
        desc_info = self.data[idx]["info"]["desc"]
        if desc_info and desc_info[0] and "channels" in desc_info[0]:
            self.rename_chs(idx=idx)
    
    def rename_chs (self, idx: int):
        """
        Rename the EEG channels returned by XDF_IMPORT with the channel names given by the XDF's metadata.

        Arguments:
            idx: The index of the stream to create info for.
        """
        original_chs_names = [i["label"][0] for i in self.data[idx]["info"]["desc"][0]["channels"][0]["channel"]]
        wrong_chs_names    = self.raw.ch_names
        assert len(original_chs_names) == len(wrong_chs_names), "Can't transfer original channels name in new mne.Raw: Lists are different in size."
        
        #Now create a remplacement dict with the original ch names as values and newly made chs names as keys
        remplacement_dic = dict(zip(wrong_chs_names, original_chs_names))
        
        self.raw.rename_channels(remplacement_dic)

    def warn_and_break (self, message: str, raise_exception: bool=True):
        """
        Print a warning message and optionally raise an exception with the input `message`.

        Arguments:
            message: The message to print and potentially use for raising an exception.
            raise_exception: Whether to raise an exception with the input `message`.
        """
        warnings.warn(message)
        if raise_exception:
            raise Exception(message)
        
    def montage_setup (self, eeg_montage=None):
        """
        Set the montage of the raw(s) using a custom mne montage label, or the path to a dig.montage file.

        Arguments:
            self: The instance of the class.
            eeg_montage: The MNE montage to set or a path to a dig.montage
        """
        if eeg_montage is not None:
            print(f"Setting '{eeg_montage}' as the montage for all EEG stream(s).")
            for eeg_stream in self.raw_all:
                try: self.raw_all[eeg_stream].set_montage(eeg_montage)
                except ValueError: raise ValueError(f"Invalid montage given to mne.set_montage(): {eeg_montage}")
        else: print("- No channels information was found. The montage can be set manually/individually by using the MNE funciton set_montage on the mne.Raw objects.")