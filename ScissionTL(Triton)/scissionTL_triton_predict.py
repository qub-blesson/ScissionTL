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
import copy


class LayerBenchmark:
    def __init__(self):
        self.model = ""
        self.name = ""
        self.input_layer = 0
        self.output_layer = 0
        self.second_prediction = -1
        self.output_size = 0
        self.tl_output_size = 0
        self.downsample_time = -1
        self.dataconvert_time = -1 




class Scenario:

    def __init__(self):
        self.device = ""  # Name of the device
        self.device_time = 0
        self.device_block = (-1, -1)
        self.device_output_size = 150000
        self.device_downsample_time = -1
        self.device_dataconvert = -1
        self.tl_output_size = 150000
        self.edge = None  # Name of edge if present
        self.edge_time = 0
        self.edge_block = (-1, -1)

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

def get_max_layer_time(result: List[LayerBenchmark] , end_index):
    #time = 0 should exclude in scenarios because it was filtered when benchmarking 
    tl_layer_time = result[end_index].downsample_time
    if tl_layer_time == -1 : 
        tl_layer_time = 1000
    return tl_layer_time


def get_data_convert_time(result: List[LayerBenchmark] , end_index):
    #time = 0 should exclude in scenarios because it was filtered when benchmarking 
    data_convert = result[end_index].dataconvert_time
    if data_convert == -1 : 
        data_convert = 1000
    return data_convert


def get_tl_output_size(result: List[LayerBenchmark] , end_index):
    tl_output_size = result[end_index].tl_output_size
    return tl_output_size



# Creates all possible scenarios across all loaded devices
def create_scenarios(application: str, devices, edges ):
    scenarios = []

    for device_result in devices:
        device_total_exec = 0 

        for index, result in enumerate(device_result):

            scenario = Scenario()
            scenario.application = application
            scenario.device = result.name

            if (index+1) == len(device_result): #last layer -> device(local) only 
                device_total_exec += result.second_prediction 

                scenario.device_time = device_total_exec 
                scenario.device_downsample_time = 0
                scenario.device_block = (0,result.output_layer)
                scenario.device_output_size = 0
                scenario.tl_output_size = 0    
                scenario.device_dataconvert = 0

                scenario.edge_time = 0
                scenario.edge_block = (-1,-1)

                device_time = device_total_exec 

                scenario.total_processing_time = device_time 

                scenario.config = "Device(" + str(scenario.device) + ") = " + str(
                    scenario.device_block[0]) + " - " + str(
                    scenario.device_block[1]) + ", Edge(" + str(scenario.edge) + ") = " + str(
                    scenario.edge_block[0]) + " - " + str(
                    scenario.edge_block[1]) 

                scenarios.append(scenario)            

            elif result.downsample_time == -1 : #the layer that can not apply transfer layer  
                device_total_exec += result.second_prediction 
                del scenario

            else:
                device_total_exec += result.second_prediction 

                scenario.device_time = device_total_exec 
                scenario.device_downsample_time = result.downsample_time
                scenario.device_block = (0,result.output_layer)
                scenario.device_output_size = result.output_size 
                scenario.tl_output_size = result.tl_output_size            
                scenario.device_dataconvert = result.dataconvert_time

                for edge_result in edges:
                    scenario = copy.deepcopy(scenario)
                    scenario.edge = edge_result[index].name
                    scenario.edge_time = edge_result[index].second_prediction
                    scenario.edge_block = (edge_result[index].input_layer,edge_result[index].output_layer)
                    if (edge_result[index].input_layer != result.output_layer+1 ):
                        print("device benchmarkings and edge benchmarkings are not matching")
                        exit()


                    device_time = device_total_exec + result.downsample_time + result.dataconvert_time

                    scenario.total_processing_time = device_time + edge_result[index].second_prediction 

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
        total_time = s.total_processing_time + get_transfer_overhead(s) + s.device_downsample_time + s.device_dataconvert
        for idx, result in enumerate(outputs[:list_count]):
            if result is None or result[0] > total_time:
                tranfer_tl_time, origin_time = transfer_tl_versus_origin(s)
                if args.versus : 
                    stats = (total_time,
                        f"tr_transfer_time: {format(round(tranfer_tl_time, 4), '.4f')} vs orignal_tranfer_time: {format(round(origin_time, 4), '.4f')} || tr_output_size: {format(round(bytes_to_megabytes(s.tl_output_size), 4), '0.4f')}MB vs original_output_size: {format(round(bytes_to_megabytes(s.device_output_size), 4), '0.4f')}MB \n (*) total time - {format(round(total_time, 4), '.4f')}s - {s.config}")
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

    elif s.edge is None :
        pass

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



# Parse Args

parser = argparse.ArgumentParser(description="Scission Prediction")

parser.add_argument('benchmark_folder', action='store', type=str,
                    help="Name of folder containing benchmark data and network statistics file")
parser.add_argument('statistics_file', action='store', type=str,
                    help="Name of network statistics file")
parser.add_argument('model', action='store', type=str, help="Name of the DNN model to predict for")

parser.add_argument('-r', dest='result_count', action='store', type=int, default=5,
                    help="Number of results to return (default: 5)")


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


application = ""

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



# Misc criteria

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

devices=[]
edges=[]

#open the benchmark data 
for filename in os.listdir(os.getcwd()):
    if not fnmatch.fnmatch(filename, f"*-*-{application}.dat"):
        continue

    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)

    full_name = filename.split(".")[0]
    system_type, name , _ = full_name.split("-")


    if system_type.upper() == "DEVICE":
        devices.append(data)
    elif system_type.upper() == "EDGE" :
        edges.append(data)

if len(devices) == 0  or len(edges) == 0:
    print("[+] No .dat benchmark files stored in benchmark_data. Exiting...")
    exit()


#Read and made the network_stats
system_stats = {}

with open(network_statistics_file, newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        system_stats[(row[0], row[1])] = NetworkStats(float(row[2]) / 1000, megabits_to_bytes(float(row[3])))



print(f"[+] {len(devices)} device, {len(edges)} edge are loaded")

scenarios = create_scenarios(application, devices, edges)

#scenarios = set(scenarios_raw)

if list_count > len(scenarios):
    list_count = len(scenarios)

s: Scenario
# Device filtering
if criteria_devices_inc:
    scenarios = [s for s in scenarios if s.device in criteria_devices_inc ]
if criteria_devices_excl:
    scenarios = [s for s in scenarios if s.device not in criteria_devices_excl ]
    
if criteria_device_layers_inc:
    scenarios = [s for s in scenarios if
                 all(x in range(s.device_block[0], s.device_block[1] + 1) for x in criteria_device_layers_inc)]
if criteria_device_layers_excl:
    scenarios = [s for s in scenarios if
                 all(x not in range(s.device_block[0], s.device_block[1] + 1) for x in criteria_device_layers_excl)]

if device_upload is not None:
    scenarios = [s for s in scenarios if s.tr_output_size <= device_upload]  
if device_time is not None:
    scenarios = [s for s in scenarios if s.device_time + s.device_dataconvert + s.device_downsample_time <= device_time]
elif device_time_percentage is not None:
    scenarios = [s for s in scenarios if (((s.device_time + s.device_dataconvert + s.device_downsample_time) / s.total_processing_time) * 100) <= device_time_percentage]
# Edge filtering
if criteria_edges_inc:
    scenarios = [s for s in scenarios if s.edge in criteria_edges_inc] 
if criteria_edges_excl:
    scenarios = [s for s in scenarios if s.edge not in criteria_edges_excl]
if criteria_edge_layers_inc:
    scenarios = [s for s in scenarios if
                 all(x in range(s.edge_block[0], s.edge_block[1] + 1) for x in criteria_edge_layers_inc)]
if criteria_edge_layers_excl:
    scenarios = [s for s in scenarios if
                 all(x not in range(s.edge_block[0], s.edge_block[1] + 1) for x in criteria_edge_layers_excl)]

if edge_time is not None:
    scenarios = [s for s in scenarios if s.edge_time <= edge_time]
elif edge_time_percentage is not None:
    scenarios = [s for s in scenarios if ((s.edge_time / s.total_processing_time) * 100) <= edge_time_percentage]


results, sorted_scenarios = get_predictions_list_execution(scenarios)

if results[0] is None:
    print("No results for the specified configuration")
    exit()

print(f"[+] {application} results")
for idx, result in enumerate(results):
    if result is not None:
        print(f"[{idx + 1}] {result[1]}")

