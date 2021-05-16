# ScissionTL Prediction
# Creates scenarios using benchmark data created from Scission Benchmark - Transfer layer version.
# Allows querying to reveal fastest scenarios
# Original Author: Luke Lockhart
# Revised: HyunHo Ahn

import argparse
import csv
import fnmatch
import os
import pickle
from enum import Enum
from typing import List, Dict


class SystemType(Enum):
    DEVICE = 1
    EDGE = 2


class LayerBenchmark:
    def __init__(self):
        self.model = ""
        self.platform = ""
        self.input_layer = 0
        self.output_layer = 0
        self.second_prediction = 0
        self.output_size = 0
        self.tl_output_size = 0
        self.downsample_time = -1
        self.upsample_time = -1
        self.dataconvert_time = -1




class System:
    def __init__(self, name: str, device_type: SystemType):
        self.name = name
        self.type = device_type
        self.bandwidth = 0
        self.ping = 0
        self.benchmarks: Dict(value, Scenario_result) = {}


class Scenario:
    global input_size

    def __init__(self):
        self.device = ""  # Name of the device
        self.device_time = 0
        self.device_block = (-1, -1)
        self.device_layers = None
        self.device_output_size = input_size
        self.device_downsample_time = -1
        self.device_dataconvert = -1
        self.tl_output_size = input_size
        self.edge = None  # Name of edge if present
        self.edge_time = 0
        self.edge_block = (-1, -1)
        self.edge_layers = None
        self.edge_output_size = 0
        self.edge_upsample_time = -1
        self.edge_dataconvert = -1
        self.total_processing_time = 0
        self.config = ""
        self.application = ""

    def __eq__(self, other):
        if not isinstance(other, Scenario):
            return NotImplemented

        return self.config == other.config

    def __hash__(self):
        return hash((self.config, self.config))


class NetworkStats:

    def __init__(self, ping, bandwidth):
        self.ping = ping
        self.bandwidth = bandwidth


# Loads pickles input file
def load_data(filename):
    try:
        with open(filename) as f:
            x = pickle.load(f)
    except:
        x = []
    return x


# Returns biggest number between two numbers
def get_biggest(x, y):
    if x > y:
        return x
    else:
        return y


# Calculates the time for a distribution to execute along with the output of that block
def get_time(result: List[LayerBenchmark], start_index, end_index):
    total_time = 0
    layers = []

    for x in range(start_index, end_index + 1):
        total_time += result[x].second_prediction
        layers.append((result[x].input_layer, result[x].output_layer, result[x].second_prediction))

    output_size = result[end_index].output_size

    return total_time, output_size, layers

def get_downsample_layer_time(result: List[LayerBenchmark] , end_index):
    #time = -1 should exclude in scenarios because it was filtered when benchmarking 
    tl_layer_time = result[end_index].downsample_time
    if tl_layer_time == -1 : 
        tl_layer_time = 1000
    return tl_layer_time



def get_upsample_layer_time(result: List[LayerBenchmark] , end_index):
    #time = -1 should exclude in scenarios because it was filtered when benchmarking 
    tl_layer_time = result[end_index].upsample_time
    if tl_layer_time == -1 : 
        tl_layer_time = 1000
    return tl_layer_time

def get_data_convert_time(result: List[LayerBenchmark] , end_index):
    #time = -1 should exclude in scenarios because it was filtered when benchmarking 
    data_convert = result[end_index].dataconvert_time
    if data_convert == -1 : 
        data_convert = 1000
    return data_convert


def get_tl_output_size(result: List[LayerBenchmark] , end_index):
    tl_output_size = result[end_index].tl_output_size
    return tl_output_size



# Creates all possible scenarios across all loaded devices
def create_scenarios(application: str, devices, edges):
    global systems
    scenarios = []

    for device in devices:
        for edge in edges:
            for x in range(len(systems[0].benchmarks[application])):

                scenario = Scenario()
                scenario.application = application

                device_time = 0
                edge_time = 0
                downsample_time = -1
                upsample_time = -1

                scenario.device = device.name

                device_time, output, device_layers = get_time(device.benchmarks[application],
                                                                0,
                                                                x)

                downsample_time = get_downsample_layer_time(device.benchmarks[application],
                                                                x)

                scenario.device_time = device_time 
                
                scenario.device_downsample_time = downsample_time

                scenario.device_layers = device_layers

                scenario.tl_output_size = get_tl_output_size(device.benchmarks[application],
                                                                x)
                
                scenario.device_dataconvert = get_data_convert_time(device.benchmarks[application],
                                                                x)
                                
                scenario.device_output_size = output 

                scenario.device_block = (device_layers[0][0], device_layers[-1][1])

                if x != len(systems[0].benchmarks[application])-1:
                    edge_time, output, edge_layers = get_time(edge.benchmarks[application],
                                                               x+1,
                                                                len(systems[0].benchmarks[application]) -1)

                    upsample_time = get_upsample_layer_time(edge.benchmarks[application],
                                                                    x)

                    scenario.edge_dataconvert = get_data_convert_time(edge.benchmarks[application],
                                                                    x)

                    scenario.edge_upsample_time = upsample_time

                                                    
                    scenario.edge = edge.name
                    scenario.edge_time = edge_time
                    scenario.edge_layers = edge_layers
                    scenario.edge_output_size = output
                    scenario.edge_block = (edge_layers[0][0], edge_layers[-1][1])

        

                total_processing_time = device_time + edge_time 
                scenario.total_processing_time = total_processing_time

                scenario.config = "Device(" + str(scenario.device) + ") = " + str(
                    scenario.device_block[0]) + " - " + str(
                    scenario.device_block[1]) + ", Edge(" + str(scenario.edge) + ") = " + str(
                    scenario.edge_block[0]) + " - " + str(
                    scenario.edge_block[1]) 

                scenarios.append(scenario)

    return scenarios


# Returns a list of results, sorted by execution time
def get_predictions_list_execution(scenarios: [Scenario]):
    global list_count
    outputs = []
    scenarios_sorted = []

    for _ in range(list_count):
        outputs.append(None)

    s: Scenario
    for s in scenarios:
        tl_time = s.device_downsample_time + s.edge_upsample_time
        tl_data_convert =  s.device_dataconvert + s.edge_dataconvert
        total_time = s.total_processing_time + get_transfer_overhead(s) + tl_time + tl_data_convert
        for idx, result in enumerate(outputs[:list_count]):
            if result is None or result[0] > total_time:
                tranfer_tl_time, origin_time = transfer_tl_versus_origin(s)
                if args.versus : 
                    stats = (total_time,
                        f"tl_layer_time: {format(round(tl_time, 4), '.4f')} + tl_dataconvert_time: {format(round(tl_data_convert, 4), '.4f')} + tl_transfer_time: {format(round(tranfer_tl_time, 4), '.4f')} vs orignal_tranfer_time: {format(round(origin_time, 4), '.4f')} || tl_output_size: {format(round(bytes_to_megabytes(s.tl_output_size), 4), '0.4f')}MB vs orig_output_size: {format(round(bytes_to_megabytes(s.device_output_size), 4), '0.4f')}MB \n (*) total time - {format(round(total_time, 4), '.4f')}s - {s.config}")
                else :
                    stats = (total_time,
                         f"{format(round(total_time, 4), '.4f')}s - {format(round(bytes_to_megabytes(s.tl_output_size), 4), '0.4f')}MB - {s.config}")
                outputs.insert(idx, stats)
                scenarios_sorted.insert(idx, s)
                break

    return outputs[:list_count], scenarios_sorted[:list_count]

def transfer_tl_versus_origin(s: Scenario):
    stats = system_stats[(s.device, s.edge)]
    tranfer_tl_time = (stats.ping + (s.tl_output_size / stats.bandwidth))
    origin_time = (stats.ping + (s.device_output_size / stats.bandwidth))
    return tranfer_tl_time, origin_time

# Calculates the transfer overhead for a scenario
def get_transfer_overhead(s: Scenario):
    transfer_overhead = 0
    filesize_to_send = s.tl_output_size
    stats: NetworkStats

    if s.edge is not None:
        stats = system_stats[(s.device, s.edge)]
        transfer_overhead += (stats.ping + (filesize_to_send / stats.bandwidth))


    elif s.edge is None:
        return transfer_overhead

    return transfer_overhead



# Calculates the transfer overhead between two devices given a file size
def get_specific_transfer_overhead(source, destination, size):
    stats: NetworkStats
    stats = system_stats[(source, destination)]

    transfer_overhead = (stats.ping + (size / stats.bandwidth))

    return transfer_overhead


def megabytes_to_bytes(x):
    return x * 1000 * 1000


def bytes_to_megabytes(x):
    return x / 1000 / 1000


def megabits_to_bytes(x):
    return round(x * 125000)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Creates a distribution graph of model over the edge pipeline

def create_graph(s: Scenario, filename):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    from pathlib import Path

    plt.rcParams.update({'font.size': 35})

    bars = []
    execution_times = []
    colors = []
    handles = []
    tl_layer_bar = []

    if s.device_layers is not None:
        handles.append(mpatches.Patch(color='dodgerblue', label='Device')) 
        execution_times.append([time[2] * 1000 for time in s.device_layers])
        tl_layer_bar.append([0 for time in s.device_layers])

        bars.append(
            [f"{result[0]}-{result[1]}" if result[0] != result[1] else f"{result[0]}" for result in s.device_layers])
        colors.append(["dodgerblue" for _ in s.device_layers])


    if s.edge_layers is not None:
        handles.append(mpatches.Patch(color='darkcyan', label='Edge'))
        handles.append(mpatches.Patch(color='yellow', label='COMM.')) 
        handles.append(mpatches.Patch(color='red', label='Tr layer')) 

        execution_times.append([get_specific_transfer_overhead(s.device, s.edge, s.tl_output_size) * 1000])
        tl_layer_bar.append([(s.device_downsample_time+s.edge_upsample_time + s.device_dataconvert + s.edge_dataconvert)*1000]) #data_convert 
        bars.append("COMM.")
        colors.append("yellow")

        execution_times.append([time[2] * 1000 for time in s.edge_layers])
        tl_layer_bar.append([0 for time in s.edge_layers])  

        bars.append(
            [f"{result[0]}-{result[1]}" if result[0] != result[1] else f"{result[0]}" for result in s.edge_layers])
        colors.append(["darkcyan" for _ in s.edge_layers])


    width = 0.8

    execution_times = np.hstack(execution_times)
    tl_layer_bar = np.hstack(tl_layer_bar)
    bars = np.hstack(bars)
    colors = np.hstack(colors)

    plt.rcParams.update({'font.size': 20})
    ind = np.arange(len(bars))
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.set_ylabel("Execution time (ms) -log scale", labelpad=10)
    ax1.bar(ind, execution_times, width, align='center', color=colors)
    ax1.bar(ind, tl_layer_bar, width, align='center', color='red' ,bottom = execution_times)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(bars)

    plt.yscale("log")
    plt.setp(ax1.get_xticklabels(), rotation=270, horizontalalignment='center')

    for axis in [ax1.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    plt.legend(handles=handles)
    file_to_open = Path(dname) / f"{filename}.png"
    plt.savefig(file_to_open, bbox_inches='tight')


# Parse Args

parser = argparse.ArgumentParser(description="Scission Prediction")

parser.add_argument('benchmark_folder', action='store', type=str,
                    help="Name of folder containing benchmark data and network statistics file")
parser.add_argument('statistics_file', action='store', type=str,
                    help="Name of network statistics file")
parser.add_argument('model', action='store', type=str, help="Name of the DNN model to predict for")
parser.add_argument('-r', dest='result_count', action='store', type=int, default=5,
                    help="Number of results to return (default: 5)")
parser.add_argument('-i', '--input', dest='input_size', action='store', type=float,
                    help="Input image filesize (MB) (default: 0.15)")
parser.add_argument('-d', '--device', dest='device_criteria', action='store', type=str, help="Device criteria")
parser.add_argument('--device-upload', dest='device_upload', action='store', type=float,
                    help="Device upload limit (MB)")
parser.add_argument('--device-time', dest='device_time', action='store', type=str,
                    help="Device time limit (s) or as a percentage of total processing time'x%%'")
parser.add_argument('-e', '--edge', dest='edge_criteria', action='store', type=str, help="Edge criteria")
parser.add_argument('--edge-time', dest='edge_time', action='store', type=str,
                    help="Edge time limit (s) or as a percentage of total processing time'x%%'")


parser.add_argument('-versus',  action='store_true', required=False,
                    help="print difference between TL and without TL (default: False) if set -versus, then it becomes true ")

args = parser.parse_args()

benchmark_folder = args.benchmark_folder
network_statistics_file = args.statistics_file

if args.result_count is not None:
    list_count = args.result_count
else:
    list_count = 5

# Parse the device specific criteria
criteria_devices_inc = []
criteria_devices_excl = []
criteria_device_layers_inc = []
criteria_device_layers_excl = []

if args.device_criteria is not None:
    for c in args.device_criteria.split(","):
        c = c.strip()

        if c.isdigit() or c == "-1": 
            criteria_device_layers_inc.append(int(c))
        elif c[0] == "*" and c[1:].isdigit() or c[1:] == "-1": # ex) *3 or *-1 
            criteria_device_layers_excl.append(int(c[1:]))
        elif c[0] == "*":
            criteria_devices_excl.append(c[1:]) #ex) *tx2 -> tx2 x 
        else:
            criteria_devices_inc.append(c) # ex) tx2 -> tx2 o

# Parse the edge specific criteria
criteria_edges_inc = []
criteria_edges_excl = []
criteria_edge_layers_inc = []
criteria_edge_layers_excl = []

if args.edge_criteria is not None:
    for c in args.edge_criteria.split(","):
        c = c.strip()

        if c.isdigit() or c == "-1":
            criteria_edge_layers_inc.append(int(c))
        elif c[0] == "*" and c[1:].isdigit() or c[1:] == "-1":
            criteria_edge_layers_excl.append(int(c[1:]))
        elif c[0] == "*":
            criteria_edges_excl.append(c[1:])
        else:
            criteria_edges_inc.append(c)

# Parse the cloud specific criteria


# Misc criteria

if args.input_size is not None:
    input_size = megabytes_to_bytes(args.input_size)
else:
    input_size = megabytes_to_bytes(0.15)

if args.device_upload is not None:
    device_upload = megabytes_to_bytes(float(args.device_upload))
else:
    device_upload = None


if args.model is not None:
    application = args.model.lower()

if args.device_time is not None:
    if args.device_time[-1] == "%":
        device_time_percentage = float(args.device_time[:-1])
        device_time = None
    else:
        device_time = float(args.device_time)
        device_time_percentage = None
else:
    device_time = None
    device_time_percentage = None

if args.edge_time is not None:
    if args.edge_time[-1] == "%":
        edge_time_percentage = float(args.edge_time[:-1])
        edge_time = None
    else:
        edge_time = float(args.edge_time)
        edge_time_percentage = None
else:
    edge_time = None
    edge_time_percentage = None


# End Parse Args

# Set path to script directory then to benchmark_data folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir(benchmark_folder)
systems = []

for filename in os.listdir(os.getcwd()):
    if not fnmatch.fnmatch(filename, "*-*.dat"):
        continue

    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)

    full_name = filename.split(".")[0]
    system_type, name = full_name.split("-")

    system_type_enum = SystemType[system_type.upper()]

    new_system = System(name, system_type_enum)
    new_system.benchmarks = data

    systems.append(new_system)

if len(systems) == 0:
    print("[+] No .dat benchmark files stored in benchmark_data. Exiting...")
    exit()

system_stats = {}
with open(network_statistics_file, newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        system_stats[(row[0], row[1])] = NetworkStats(float(row[2]) / 1000, megabits_to_bytes(float(row[3])))

devices = [d for d in systems if d.type == SystemType.DEVICE]
edges = [d for d in systems if d.type == SystemType.EDGE]

print(f"[+] {len(systems)} systems loaded : {len(devices)} device, {len(edges)} edge")

scenarios_raw = create_scenarios(application, devices, edges)
scenarios = set(scenarios_raw)

if list_count > len(scenarios):
    list_count = len(scenarios)

s: Scenario
# Device filtering
if criteria_devices_inc:
    scenarios = [s for s in scenarios if s.device in criteria_devices_inc and s.device_layers is not None]
if criteria_devices_excl:
    scenarios = [s for s in scenarios if s.device not in criteria_devices_excl and s.device_layers is not None]
if criteria_device_layers_inc:
    scenarios = [s for s in scenarios if
                 all(x in range(s.device_block[0], s.device_block[1] + 1) for x in criteria_device_layers_inc)]
if criteria_device_layers_excl:
    scenarios = [s for s in scenarios if
                 all(x not in range(s.device_block[0], s.device_block[1] + 1) for x in criteria_device_layers_excl)]
if device_upload is not None:
    scenarios = [s for s in scenarios if s.tl_output_size <= device_upload]  
if device_time is not None:
    scenarios = [s for s in scenarios if s.device_time + s.device_dataconvert + s.device_downsample_time <= device_time]
elif device_time_percentage is not None:
    scenarios = [s for s in scenarios if (((s.device_time + s.device_dataconvert + s.device_downsample_time) / s.total_processing_time) * 100) <= device_time_percentage]

# Edge filtering
if criteria_edges_inc:
    scenarios = [s for s in scenarios if s.edge in criteria_edges_inc and s.edge_layers is not None]
if criteria_edges_excl:
    scenarios = [s for s in scenarios if s.edge not in criteria_edges_excl and s.edge_layers is not None]
if criteria_edge_layers_inc:
    scenarios = [s for s in scenarios if
                 all(x in range(s.edge_block[0], s.edge_block[1] + 1) for x in criteria_edge_layers_inc)]
if criteria_edge_layers_excl:
    scenarios = [s for s in scenarios if
                 all(x not in range(s.edge_block[0], s.edge_block[1] + 1) for x in criteria_edge_layers_excl)]
if edge_time is not None:
    scenarios = [s for s in scenarios if s.edge_time + s.edge_dataconvert + s.edge_upsample_time <= edge_time]
elif edge_time_percentage is not None:
    scenarios = [s for s in scenarios if (((s.edge_time + s.edge_dataconvert + s.edge_upsample_time) / s.total_processing_time) * 100) <= edge_time_percentage]


results, sorted_scenarios = get_predictions_list_execution(scenarios)

if results[0] is None:
    print("No results for the specified configuration")
    exit()

print(f"[+] {application} results")
for idx, result in enumerate(results):
    if result is not None:
        print(f"[{idx + 1}] {result[1]}")

create_graph(sorted_scenarios[0], f"{application}-{round(results[0][0], 2)}s")
print(f"[+] Graph created: {application}-{round(results[0][0], 2)}s")
