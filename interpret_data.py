# For creating a relative path for importing files, if required.
# import sys
# sys.path.append("your_path")

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aston.tracefile import TraceFile
from aston.tracefile.agilent_uv import *

# export PYTHONPATH=$PYTHONPATH:/Users/EricShen/Desktop/Aston
# python3 -c 'import pandas'

# Replace the below with your custom files and strings.
RAW_DATA_PATH = ""
UV_EXTENSION = ""
CSV_PATH = ""
WAVELENGTH = 210.


def test_open_data():
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f, delimiter=',')
        print(reader.fieldnames)
        row_count = 0
        for row in reader:
            row_count += 1
        print(row_count)
        for row in reader:
            print(', '.join(row))

    df = TraceFile(RAW_DATA_PATH + UV_EXTENSION)
    print(df.info)
    data = df.data
    print(type(data))
    print(data.shape)
    print(data.values)


def get_data(plot=False,
             export_txt=False,
             export_csv=False,
             export_numpy=False,
             export_pandas=False):
    df = AgilentCSDAD2(RAW_DATA_PATH + UV_EXTENSION)

    chr = df.data
    traces = chr.traces

    wavelength_ind = chr.columns.index(WAVELENGTH)
    wavelength_trace = traces[wavelength_ind]

    if (plot):
        wavelength_trace.plot()

    ri_values = chr.values[:, -1]
    num_traces = ri_values.shape[0]
    f, csvfile, writer, npdata, df = None, None, None, None, None
    col_titles = ['RT(milliseconds)',
                  'RT(minutes) - NOT USED BY IMPORT', 'RI', '210']

    if (export_txt):
        f = open("test_data.txt", "w+")
        f.write(",".join(col_titles) + "\n")

    if (export_numpy):
        npdata = np.empty((num_traces, 4))

    if (export_pandas):
        df = pd.DataFrame(None, columns=col_titles)

    if (export_csv):
        csvfile = open("test_data.csv", "w+")
        writer = csv.DictWriter(csvfile, fieldnames=col_titles)
        writer.writeheader()

    for i in range(len(wavelength_trace)):
        ms = wavelength_trace.index[i]
        minutes = ms / 60000.
        val = wavelength_trace[i]
        ri = ri_values[i]
        row = {
            col_titles[0]: int(ms),
            col_titles[1]: minutes,
            col_titles[2]: ri,
            col_titles[3]: val
        }
        if (export_txt):
            f.write(str(int(ms)) + "," + str(minutes) +
                    ",{:.1f}".format(ri) + ",{:.1f}".format(val) + "\n")
        if (export_csv):
            writer.writerow(row)
        if (export_numpy):
            npdata[i] = np.array([int(ms), minutes, ri, val])
        if (export_pandas):
            df = df.append(row, ignore_index=True)

    if (export_txt):
        f.close()

    if (export_csv):
        csvfile.close()

    ret_val = []
    if (export_numpy):
        ret_val.append(npdata)
    if (export_pandas):
        ret_val.append(df)
    return ret_val


def check_data(result, target, write_data=False, plot=False):
    if (plot):
        plt.plot(result[:, 0], result[:, -1], label="Result", alpha=0.4)
        plt.plot(target[:, 0], target[:, -1], label="Target", alpha=0.4)
        plt.title("Result vs. Target Chromatograms")
        plt.xlabel("Milliseconds")
        plt.ylabel("210 nm")
        plt.legend()
        plt.show()
        # peaks seem to be offset by 100ms
    f = None
    if (write_data):
        f = open("errors.txt", "w+")
    sum_percentage_error = 0.
    n = min(result.shape[0], target.shape[0])
    for i in range(1, n):
        result_val, target_val = result[i, -1], target[i, -1]
        percentage_error = (target_val - result_val) / target_val
        if (write_data):
            f.write("{:.2%}".format(percentage_error) + "\n")
        sum_percentage_error += abs(percentage_error)
    if (write_data):
        f.close()
    return sum_percentage_error / n


example_data = np.genfromtxt(CSV_PATH, delimiter=",")
actual_data = get_data(export_numpy=True, export_csv=True)[0]
check_data(actual_data, example_data, plot=True)
