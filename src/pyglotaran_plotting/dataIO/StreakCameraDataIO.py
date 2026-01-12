"""Module containing the CSV Data IO plugin."""

from __future__ import annotations

import os.path
import re
from enum import Enum
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from glotaran.io import DataIoInterface
from glotaran.io import register_data_io
from glotaran.io.prepare_dataset import prepare_time_trace_dataset



class DataFileType(Enum):
    time_explicit = "Time explicit"
    wavelength_explicit = "Wavelength explicit"

class StreakExplicitFile:
    """
    Abstract class for loading streakcamera data in DAC format. 
    Applies Resampling of the data and spectral correction.
    """
    def __init__(self, filepath: str | None, dataset: xr.DataArray | None = None):
        self._file_data_format = None
        self._data = []
        self._times = []
        self._spectral_indicies = []
        self._label = ""
        self._comment = ""
        absfilepath = os.path.realpath(filepath)
        if dataset is not None:
            self._data = np.array(dataset.values).T
            self._times = np.array(dataset.coords["time"])
            self._spectral_indices = np.array(dataset.coords["spectral"])
            self._file = filepath
        elif os.path.isfile(filepath):
            self._file = filepath
        elif os.path.isfile(absfilepath):
            self._file = absfilepath
        else:
            raise Exception(f"Path does not exist: {filepath}, {absfilepath}")
    
    def write(
            self,
            overwrite=False,
            comment="",
            file_format=DataFileType.wavelength_explicit,
            number_format="%.10e",
            ):
        if os.path.isfile(self._file) and not overwrite:
            raise FileExistsError(f"File already exist:\n{self._file}")
        comment = f"{self._comment} {comment}"
        comments = f"# Filename: {str(self._file)}\n{' '.join(comment.splitlines())}\n"
        if file_format == DataFileType.wavelength_explicit:
            wav_tmp = ",".join(repr(num) for num in self._spectral_indices)
            wav = "ps|nm" + wav_tmp
            raw_data = np.vstack((self._times.T, self._data)).T
        else:
            raise NotImplementedError
        
        np.savetxt(
            self._file,
            raw_data,
            fmt=number_format,
            delimiter=",",
            newline="\n",
            header="",
            footer="",
            comments="",
        )

    def read(self, prepare: bool = True) -> xr.Dataset:        
        if not os.path.isfile(self._file):
            raise FileNotFoundError("File does not exist.")
        
        ## Hardcode the file data format as it is always the same
        self._file_data_format = DataFileType.wavelength_explicit

        # Read data from file
        tmp_wv = pd.read_csv(self._file, sep='\t', nrows=2, header=None)
        # resample data due to artifacts
        wv_axis = tmp_wv.iloc[0,1:].apply(pd.to_numeric, errors='coerce').values
        # wv_axis = wv_axis.iloc[0].to_numpy() 

        wv_axis = wv_axis[::2]
        tmp_data = pd.read_csv(self._file, skiprows=1, sep='\t', header=None, dtype=np.float64).values
        times = tmp_data[:,0]
        data_arr = tmp_data[:,1:]
        data_arr = data_arr[:,::2]     
                 
        # Write to class
        self._spectral_indices = np.flip(wv_axis)
        self._times = times
        self._data = np.fliplr(data_arr)
        return self.dataset(prepare=prepare)


    def dataset(self, prepare: bool = True) -> xr.Dataset | xr.DataArray:
        data = self._data
        if self._file_data_format == DataFileType.time_explicit:
            data = data.T
        dataset = xr.DataArray(
            data, coords=[("time", self._times), ("spectral", self._spectral_indices)]
        )
        if prepare:
            dataset = prepare_time_trace_dataset(dataset)
        return dataset

class WavelengthExplicitFile(StreakExplicitFile):
    """
    Represents a wavelength explicit file
    """

    def get_explicit_axis(self):
        return self._spectral_indices

    def get_secondary_axis(self):
        return self.observations()

    def get_data_row(self, index):
        return []

    def add_data_row(self, row):
        if self._timepoints is None:
            self._timepoints = []
        self._timepoints.append(float(row.pop(0)))

        if self._spectra is None:
            self._spectra = []
        self._spectra.append(float(row))

    def get_format_name(self):
        return DataFileType.wavelength_explicit

    def times(self):
        return self.get_secondary_axis()

    def wavelengths(self):
        return self.get_explicit_axis()

@register_data_io(["dac"])
class StreakCameraDataIO(DataIoInterface):
    def load_dataset(self, file_name: str, *, prepare: bool = True):
        """Plugin for loading Combined CSV datasets exported from matlab.

        Parameters
        ----------
        fname : str
            Name of the dac file.

        Returns
        -------
        dataset : xr.Dataset

        Notes
        -----
        .. [1] https://glotaran.github.io/legacy/file_formats
        """

        data_file = (
            WavelengthExplicitFile(filepath=file_name)
        )

        return data_file.read(prepare=prepare)