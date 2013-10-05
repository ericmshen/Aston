# -*- coding: utf-8 -*-

#    Copyright 2011-2013 Roderick Bovee
#
#    This file is part of Aston.
#
#    Aston is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Aston is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Aston.  If not, see <http://www.gnu.org/licenses/>.


"""
Core functions for dealing with TimeSeries.
"""

import json
import zlib
import struct
try:  # preliminary support for pypy
    import numpypy as np
except:
    import numpy as np


class TimeSeries(object):
    """
    A TimeSeries is any set of observations (self.data) taken at
    definite time points (self.time) of certain types (self.ions).
    These types can include m/z's for mass spec data, wavelengths
    for UV/Visible data, string descriptors for "actuals" data
    (like Presure/Temperature/etc).

    Interally, all times are assumed to be in minutes elapsed
    since start of experiment (although an internal "x-units"
    variable may be added at some point).

    Self.data may be a two-dimensional array or a sparse matrix
    depending on which holds the data more space efficiently.

    Many functions in TimeSeries can take a time window (twin)
    argument which only selects a certain x-range to perform
    that function on.
    """
    def __init__(self, data, times, ions=[]):
        if data is not None:
            if len(data.shape) == 1:
                data = np.atleast_2d(data).T
                if ions == []:
                    ions = ['']
            assert times.shape[0] == data.shape[0]
            assert len(ions) == data.shape[1]
            try:  # python 2
                assert all(isinstance(i, basestring) for i in ions)
            except:  # python 3
                assert all(isinstance(i, str) for i in ions)
        self._rawdata = data
        self.times = times
        self.ions = ions

    def _slice_idxs(self, twin=None):
        """
        Returns a slice of the incoming array filtered between
        the two times specified. Assumes the array is the same
        length as self.data. Acts in the time() and trace() functions.
        """
        if twin is None:
            return 0, self._rawdata.shape[0]

        tme = self.times.copy()

        if twin[0] is None:
            st_idx = 0
        else:
            st_idx = (np.abs(tme - twin[0])).argmin()
        if twin[1] is None:
            en_idx = self._rawdata.shape[0]
        else:
            en_idx = (np.abs(tme - twin[1])).argmin() + 1
        return st_idx, en_idx

    def time(self, twin=None):
        """
        Returns an array with all of the time points at which
        data was collected
        """
        st_idx, en_idx = self._slice_idxs(twin)
        tme = self.times[st_idx:en_idx].copy()
        #return the time series
        return tme

    def len(self, twin=None):
        st_idx, en_idx = self._slice_idxs(twin)
        return en_idx - st_idx

    def twin(self, twin):
        st_idx, en_idx = self._slice_idxs(twin)
        return TimeSeries(self._rawdata[st_idx:en_idx], \
                          self.times[st_idx:en_idx], self.ions)

    def trace(self, val='TIC', tol=0.5, twin=None):
        st_idx, en_idx = self._slice_idxs(twin)

        if isinstance(val, (int, float, np.float32, np.float64)):
            val = str(val)

        if val == 'TIC' and 'TIC' not in self.ions:
            # if a TIC is being requested and we don't have
            # a prebuilt one, sum up the axes
            data = self._rawdata[st_idx:en_idx, :].sum(axis=1)
            #TODO: this fails for sparse matrices?
            #data = np.array(self._rawdata[st_idx:en_idx, :].sum(axis=0).T)[0]
        elif val == '!':
            # this is for peaks, where we return the first
            # ion by default; shouldn't be accessible from the
            # ions dialog box because !'s are stripped out
            data = self._rawdata[st_idx:en_idx, 0]
            val = str(self.ions[0])
        else:
            is_num = lambda x: set(x).issubset('1234567890.')
            if is_num(val):
                ions = np.array([float(i) if is_num(i) else np.nan \
                                 for i in self.ions])
                rows = np.where(np.abs(ions - float(val)) < tol)[0]
            elif val in self.ions:
                rows = np.array([self.ions.index(val)])
            else:
                rows = []

            # if no rows, return an array of NANs
            # otherwise, return the data
            if len(rows) == 0:
                data = np.zeros(en_idx - st_idx) * np.nan
            else:
                data = self._rawdata[st_idx:en_idx, rows].sum(axis=1)
        return TimeSeries(data, self.times[st_idx:en_idx], [val])

    def scan(self, time, to_time=None):
        """
        Returns the spectrum from a specific time.
        """
        idx = (np.abs(self.times - time)).argmin()
        if to_time is None:
            if type(self._rawdata) == np.ndarray:
                ion_abs = self._rawdata[idx, :].copy()
            else:
                ion_abs = self._rawdata[idx, :].astype(float).toarray()[0]
            #return np.array(self.ions), ion_abs
            return np.vstack([np.array([float(i) for i in self.ions]), \
              ion_abs])
        else:
            #TODO: should there be an option to average instead of summing?
            en_idx = (np.abs(self.times - to_time)).argmin()
            idx, en_idx = min(idx, en_idx), max(idx, en_idx)
            if type(self._rawdata) == np.ndarray:
                ion_abs = self._rawdata[idx:en_idx + 1, :].copy()
            else:
                ion_abs = self._rawdata[idx:en_idx + 1, :]
                ion_abs = ion_abs.astype(float).toarray()
            return np.vstack([np.array([float(i) for i in self.ions]), \
              ion_abs.sum(axis=0)])

    def get_point(self, trace, time):
        """
        Return the value of the trace at a certain time.

        This has the advantage of interpolating the value
        if the time is not exact.
        """
        from scipy.interpolate import interp1d
        ts = self.trace(trace)
        f = interp1d(ts.times, ts.data.T, \
          bounds_error=False, fill_value=0.0)
        return f(time)[0]

    def as_2D(self):
        """
        Returns two matrices, one of the data and the other of
        the times and trace corresponding to that data.

        Useful for making two-dimensional "heat" plots.
        """
        if self.times.shape[0] == 0:
            return (0, 1, 0, 1), np.array([[0]])
        ions = [float(i) for i in self.ions]
        ext = (self.times[0], self.times[-1], min(ions), max(ions))
        if type(self._rawdata) == np.ndarray:
            grid = self._rawdata[:, np.argsort(self.ions)].transpose()
        else:
            from scipy.sparse import coo_matrix
            data = self._rawdata[:, 1:].tocoo()
            data_ions = np.array([float(self.ions[i]) for i in data.col])
            grid = coo_matrix((data.data, (data_ions, data.row))).toarray()
        return ext, grid

    def as_text(self, width=80, height=20):
        raise NotImplementedError

    def as_sound(self, speed=60, cutoff=50):
        import scipy.io.wavfile
        import scipy.signal

        # make a 1d array for the sound
        to_t = lambda t: (t - self.times[0]) / speed
        wav_len = int(to_t(self.times[-1]) * 60 * 44100)
        wav = np.zeros(wav_len)

        # create an artificial array to interpolate times out of
        tmask = np.linspace(0, 1, len(self.times))

        # come up with a mapping from mz to tone
        min_hz, max_hz = 50, 1000
        is_num = lambda x: set(x).issubset('1234567890.')
        min_mz = min(float(i) if is_num(i) else np.inf for i in self.ions)
        max_mz = max(float(i) if is_num(i) else 0 for i in self.ions)

        def mz_to_wv(mz):
            """
            Maps a wavelength/mz to a tone.
            """
            try:
                mz = float(mz)
            except:
                return 100
            wv = (mz * (max_hz - min_hz) - max_hz * min_mz + min_hz * max_mz) \
                    / (max_mz - min_mz)
            return int(44100 / wv)

        # go through each trace and map it into the sound array
        for i, mz in enumerate(self.ions):
            if float(mz) < cutoff:
                # clip out mz/wv below a certain threshold
                # handy if data has low level noise
                continue
            print(str(i) + '/' + str(len(self.ions)))
            inter_x = np.linspace(0, 1, wav[::mz_to_wv(mz)].shape[0])
            wav[::mz_to_wv(mz)] += np.interp(inter_x, tmask, self.data[:, i])

        # scale the new array and write it out
        scaled = wav / np.max(np.abs(wav))
        scaled = scipy.signal.fftconvolve(scaled, np.ones(5) / 5, mode='same')
        scaled = np.int16(scaled * 32767)
        scipy.io.wavfile.write('test.wav', 44100, scaled)

    def plot(self, show=False):
        """
        Plots the top trace in matplotlib.  Useful for data exploration on
        the commandline; not used in the PyQt gui.
        """
        import matplotlib.pyplot as plt
        plt.plot(self.times, self.y)
        if show:
            plt.show()

    def retime(self, new_times):
        return TimeSeries(self._retime(new_times), new_times, self.ions)

    def _retime(self, new_times, fill=0.0):
        from scipy.interpolate import interp1d
        if new_times.shape == self.times.shape:
            if np.all(np.equal(new_times, self.times)):
                return self._rawdata
        f = lambda d: interp1d(self.times, d, \
            bounds_error=False, fill_value=fill)(new_times)
        return np.apply_along_axis(f, 0, self._rawdata)

    def adjust_time(self, offset=0.0, scale=1.0):
        t = scale * self.times + offset
        return TimeSeries(self._rawdata, t, self.ions)

    def has_ion(self, ion):
        if ion in self.ions:
            return True
        return False

    def _apply_data(self, f, ts):
        """
        Convenience function for all of the math stuff.
        """
        if type(ts) == int or type(ts) == float:
            d = ts * np.ones(self._rawdata.shape[0])
        elif ts is None:
            d = None
        elif all(ts.times == self.times):
            d = ts.data[:, 0]
        else:
            d = ts._retime(self.times)[:, 0]

        new_data = np.apply_along_axis(f, 0, self.data, d)
        return TimeSeries(new_data, self.times, self.ions)

    def __add__(self, ts):
        return self._apply_data(lambda x, y: x + y, ts)

    def __sub__(self, ts):
        return self._apply_data(lambda x, y: x - y, ts)

    def __mul__(self, ts):
        return self._apply_data(lambda x, y: x * y, ts)

    def __div__(self, ts):
        return self._apply_data(lambda x, y: x / y, ts)

    def __truediv__(self, ts):
        return self.__div__(ts)

    def __reversed(self):
        raise NotImplementedError

    def __iadd__(self, ts):
        return self._apply_data(lambda x, y: x + y, ts)

    def __isub__(self, ts):
        return self._apply_data(lambda x, y: x - y, ts)

    def __imul__(self, ts):
        return self._apply_data(lambda x, y: x * y, ts)

    def __idiv__(self, ts):
        return self._apply_data(lambda x, y: x / y, ts)

    def __neg__(self):
        return self._apply_data(lambda x, y: -x, None)

    def __abs__(self):
        return self._apply_data(lambda x, y: abs(x), None)

    def __and__(self, ts):
        """
        Merge together two TimeSeries.
        """
        if ts is None:
            return self
        t_step = self.times[1] - self.times[0]
        b_time = min(self.times[0], ts.times[0] + \
                     (self.times[0] - ts.times[0]) % t_step)
        e_time = max(self.times[-1], ts.times[-1] + t_step - \
                     (ts.times[-1] - self.times[-1]) % t_step)
        t = np.arange(b_time, e_time, t_step)
        x0 = self._retime(t, fill=np.nan)
        x1 = ts._retime(t, fill=np.nan)
        data = np.hstack([x0, x1])
        ions = self.ions + ts.ions
        ts = TimeSeries(data, t, ions)
        return ts

    @property
    def y(self):
        #TODO: should be defined in terms of self.trace('!') ?
        return self.data.T[0]

    @property
    def data(self):
        if type(self._rawdata) == np.ndarray:
            return self._rawdata
        elif type(self._rawdata) == np.matrix:
            #TODO: something is initializing me with a matrix?
            # happens somewhere in the sparse decomposition code
            return self._rawdata.A
        else:
            return self._rawdata.astype(float).toarray()

    def compress(self):
        d = self.data.tostring()
        t = self.times.astype(float).tostring()
        lt = struct.pack('<L', len(t))
        i = json.dumps(self.ions).encode('utf-8')
        li = struct.pack('<L', len(i))
        try:  # python 2
            return buffer(zlib.compress(li + lt + i + t + d))
        except NameError:  # python 3
            return zlib.compress(li + lt + i + t + d)


def decompress_to_ts(zdata):
    data = zlib.decompress(zdata)
    li = struct.unpack('<L', data[0:4])[0]
    lt = struct.unpack('<L', data[4:8])[0]
    i = json.loads(data[8:8 + li].decode('utf-8'))
    t = np.fromstring(data[8 + li:8 + li + lt])
    d = np.fromstring(data[8 + li + lt:])

    return TimeSeries(d.reshape(len(t), len(i)), t, i)


def ts_func(f):
    """
    This wraps a function that would normally only accept an array
    and allows it to operate on a TimeSeries. Useful for applying
    numpy functions to TimeSeries.
    """
    def wrap_func(ts, *args):
        return TimeSeries(f(ts.y, *args), ts.times)
    return wrap_func