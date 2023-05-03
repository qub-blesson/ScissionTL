# ScissionTL Benchmark
# Benchmarks keras DNNs to create layer profiles, output to benchmark data file to be used with Scission Predict - Implementation version for transfer layer and triton.
# Original Author: Luke Lockhart
# Revised: Hyunho Ahn

import argparse
import os
import pickle
import time
from collections.abc import Iterable
from pathlib import Path
import csv
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inceptionresnetv2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetlarge
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetmobile

from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.resnet import ResNet101

from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.models import Sequential


class ModelResult: # using when normal execution 
    def __init__(self):
        self.model = ""
        self.layer_count = 0
        self.load_time = 0
        self.preprocess_time = 0
        self.first_prediction = 0
        self.second_prediction = 0


class LayerBenchmark: # using when layer benchmark 
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

# Recursively gets the output of a layer, used to build up a submodel
def get_output_of_layer(layer, new_input, starting_layer_name):
    global layer_outputs
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    if layer.name == starting_layer_name:
        out = layer(new_input)
        layer_outputs[layer.name] = out
        return out

    prev_layers = []
    for node in layer._inbound_nodes:
        if isinstance(node.inbound_layers, Iterable):
            prev_layers.extend(node.inbound_layers)
        else:
            prev_layers.append(node.inbound_layers)

    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl, new_input, starting_layer_name)])

    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out


# Returns a submodel for a specified input and output layer
def get_model(input_layer: int, output_layer: int):
    global selected_model

    layer_number = input_layer
    starting_layer_name = selected_model.layers[layer_number].name

    if input_layer == 0:
        new_input = selected_model.input

        return models.Model(new_input, selected_model.layers[output_layer].output)
    else:
        new_input = layers.Input(batch_shape=selected_model.get_layer(starting_layer_name).get_input_shape_at(0))

    new_output = get_output_of_layer(selected_model.layers[output_layer], new_input, starting_layer_name)
    model = models.Model(new_input, new_output)

    return model


# Navigates the model structure to find regions without parallel paths, returns valid split locations
def create_valid_splits():
    global selected_model
    model = selected_model

    layer_index = 1
    multi_output_count = 0

    valid_splits = []
    for layer in model.layers[1:]:

        if len(layer._outbound_nodes) > 1:
            multi_output_count += len(layer._outbound_nodes) - 1

        if type(layer._inbound_nodes[0].inbound_layers) == list:
            if len(layer._inbound_nodes[0].inbound_layers) > 1:
                multi_output_count -= (
                        len(layer._inbound_nodes[0].inbound_layers) - 1)

        if multi_output_count == 0:
            valid_splits.append(layer_index)

        layer_index += 1

    return valid_splits

# Pre-processes the input image for a specific model
def process_input(application):
    global batch_size 
    print(batch_size)
    if application == "xception":
        image = np.random.rand(batch_size, 299,299,3)
        image = preprocess_input_vgg16(image)
    elif application == "vgg16":
        image = np.random.rand(batch_size, 224,224,3) 
        image = preprocess_input_vgg16(image)
    elif application == "vgg19":
        image = np.random.rand(batch_size, 224,224,3) 
        image = preprocess_input_vgg19(image)
    elif application == "resnet50" or application == "resnet101" or application == "resnet152":
        image = np.random.rand(batch_size, 224,224,3)
        image = preprocess_input_resnet(image)
    elif application == "resnet50v2" or application == "resnet101v2" or application == "resnet152v2":
        image = np.random.rand(batch_size, 224,224,3)
        image = preprocess_input_resnetV2(image)
    elif application == "inception_v3":
        image = np.random.rand(batch_size, 299,299,3)
        image = preprocess_input_inceptionv3(image)
    elif application == "inceptionresnet_v2":
        image = np.random.rand(batch_size, 299,299,3)
        image = preprocess_input_inceptionresnetv2(image)
    elif application == "mobilenet":
        image = np.random.rand(batch_size, 224,224,3) 
        image = preprocess_input_mobilenet(image)
    elif application == "mobilenetv2":
        image = np.random.rand(batch_size, 224,224,3)
        image = preprocess_input_vgg16(image)
    elif application == "densenet121" or application == "densenet169" or application == "densenet201":
        image = np.random.rand(batch_size, 224,224,3)
        image = preprocess_input_densenet(image)
    elif application == "nasnetlarge":
        image = np.random.rand(batch_size, 331,331,3)
        image = preprocess_input_nasnetlarge(image)
    elif application == "nasnetmobile":
        image = np.random.rand(batch_size, 224,224,3)
        image = preprocess_input_nasnetmobile(image)
    else:
        print("[+] Application not found.")

    return image


# Benchmarks the normal execution of a DNN - entire model unmodified
def benchmark_normal_execution(selected_model, application):
    global normal_result
    global number_of_repeats

    global reference_prediction

    normal_result.model = application
    normal_result.layer_count = len(selected_model.layers)

    print("[-] Benchmarking normal execution", end='', flush=True)

    start_preprocess = time.time()
    image = process_input(application)
    total_preprocess = time.time() - start_preprocess
    normal_result.preprocess_time = total_preprocess

    start_first = time.time()
    selected_model(image,training=False) 
    total_first = time.time() - start_first
    normal_result.first_prediction = total_first

    start_second = time.time()

    for _ in range(number_of_repeats):
        reference_prediction = selected_model(image,training=False)

    total_second = time.time() - start_second
    
    normal_result.second_prediction = (total_second / number_of_repeats)
    

def valid_tl_point_checker(input_image):
    if (tf.rank(tf.squeeze(input_image, [0])) != 3): # 4-D
        return False
  
    if(input_image.shape[1] == 1 and input_image.shape[2] == 1  ):  # (None,1,1,1024) x 
        return False
     
    return True

# Benchmarks the indiviual layers/blocks of a DNN
def benchmark_individual_execution(application):
    global individual_results
    global number_of_repeats
    global split_points
    global reference_prediction
    global device_name

    print("[-] Benchmarking individual layers", end='', flush=True)

    image = process_input(application)

    previous_input = image

    for index, split_point in enumerate(split_points):

        result = LayerBenchmark()
        result.model = application
        result.name = device_name
        if index == 0:
            first_point = 0
        else:
            first_point = split_points[index - 1] + 1

        result.input_layer = first_point
        result.output_layer = split_point

        new_model = get_model(first_point, split_point)

        output = previous_input
        previous_input = new_model(previous_input,training=False)

        start_second = time.time()

        for x in range(number_of_repeats):
            output_final = new_model(output, training=False)
            
        end_second = time.time() - start_second
        result.second_prediction = end_second / number_of_repeats
        
        if valid_tl_point_checker(output_final) == False: 
        
            result.upsample_time = -1
            result.downsample_time = -1
            result.tl_output_size = 0   

        else:
            
            downsample_layers = Sequential([
                layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid',input_shape= tf.shape(tf.squeeze(output_final, [0])))
            ])
            tl_downsample_output = downsample_layers(output_final,training=False)
            
            start_second = time.time()
            for x in range(number_of_repeats):
                tl_downsample_output = downsample_layers(output_final,training=False)

            end_second = time.time() - start_second
            result.downsample_time = end_second / number_of_repeats

            dataconvert = tl_downsample_output.numpy()
            dataconvert.tobytes()
            start_second = time.time()
            for x in range(number_of_repeats):
                dataconvert = tl_downsample_output.numpy()
                dataconvert.tobytes()
            end_second = time.time() - start_second
            result.dataconvert_time = end_second / number_of_repeats
            
            np.save("tl_size", tl_downsample_output)
            result.tl_output_size = os.stat("tl_size.npy").st_size

        
        np.save("fsize", output_final)
        result.output_size = os.stat("fsize.npy").st_size

        individual_results.append(result)

        del result



    #if reference_prediction.all() != output_final.all():
    if reference_prediction.numpy().all() != output_final.numpy().all():

        print("[WARNING] PREDICATION ACCURACY DIFFERS FROM NORMAL EXECUTION [WARNING]")


# Pickles and saves the benchmark data
def save_data(data, device_type, device_name, application):
    with open(f"{device_type}-{device_name}-{application}.dat", "wb") as f:
        pickle.dump(data, f)
    


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


model_dict = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "xception": Xception,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnet50v2": ResNet50V2,
    "resnet101v2": ResNet101V2,
    "resnet152v2": ResNet152V2,
    "inception_v3": InceptionV3,
    "inceptionresnet_v2": InceptionResNetV2,
    "mobilenet": MobileNet,
    "mobilenetv2": MobileNetV2,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "nasnetlarge": NASNetLarge,
    "nasnetmobile": NASNetMobile 
}

# Script Start

# Parse Args

parser = argparse.ArgumentParser(description="ScissionTL benchmarking for Keras Models")

parser.add_argument('name', action='store', type=str, help="Platform Name (e.g. TX2_GPU)")
parser.add_argument('model',  type=str, action='store',
                    help="model name(e.g densenet169)")
parser.add_argument('-batch', action='store', type=int, help="Batch size", required=False, default=1)
parser.add_argument('-r', '-repeats', dest='repeats', action='store', type=int, required=False,
                    help="Number of repeats for averaging (default: 10)")
parser.add_argument('-dc', '--disablecuda', dest='cuda', action='store', type=str, required=False,
                    help="Disable cuda (default: False)")
parser.add_argument('-server',  type=str, required=True,
                    help='Server name or url:port')


args = parser.parse_args()

core = "gpu" 


if args.server == "ampere":
    server = '192.169.0.1:8001'
else:
    server = args.server


if args.repeats is not None:
    number_of_repeats = args.repeats
else:
    number_of_repeats = 10
    
if args.cuda is not None:
    disable_cuda = str2bool(args.cuda)
else:    
    disable_cuda = False

if disable_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    core = "cpu"

device_type = "device"
device_name = args.name
batch_size = args.batch
application = args.model


# End Parse Args

start_entire = time.time()

individual_results = []
current_path = os.getcwd()

reference_prediction = None

#script start
start_application = time.time()

normal_result = ModelResult()

# Loading Model Start
start_load = time.time()
selected_model = model_dict[application]()
total_load = time.time() - start_load
normal_result.load_time = total_load
# Load Model End

layer_outputs = {}
split_points = create_valid_splits()

print("[+] " + application + " - " + " Layers: " + str(len(selected_model.layers)) + " - Split Points: " + str(
    len(create_valid_splits()) - 1) + " - Loading took: " + str(total_load))

# Normal Execution Benchmark Start
start_normal = time.time()
benchmark_normal_execution(selected_model, application)
total_normal = time.time() - start_normal

print(f" - {total_normal}")
# Normal Execution Benchmark End

# Individual Layer Benchmark Start
start_individual = time.time()
benchmark_individual_execution(application)
total_individual = time.time() - start_individual

print(f" - {total_individual}")
# Individual Layer Benchmark End

# Calculate difference between normal execution and the summation of individual layers
total_exec = 0
for result in individual_results:
    total_exec += result.second_prediction

extra = ((total_exec / normal_result.second_prediction) * 100) - 100
print(f"[-] NE: {normal_result.second_prediction} - SUM: {total_exec} - % Change: {extra}")


total_application = time.time() - start_application
print(f"[+] Benchmarking {application} took: {total_application} \n")

K.clear_session()

save_data(individual_results, device_type, device_name, application)

total_entire = time.time() - start_entire
print(f"[+] Benchmarking took: {total_entire}")


# Device benchmark Script End

# Edge benchmakr Script Start
print(current_path)

edge_results = []

for index, split_point in enumerate(split_points):
    
    if split_point == split_points[-1]:
        print("last -> exit")
        break


    result = LayerBenchmark()
    result.model = application
    result.name = args.server
    
    result.input_layer = split_point+1
    result.output_layer = split_points[-1]   

    if valid_tl_point_checker(selected_model.layers[split_point].output):
        
        csv_file_path = current_path + f"/{application}_{str(index)}_profile.csv"

        if os.path.exists(csv_file_path) : 
            os.system(f"rm {csv_file_path}")
            print("rm the file")

        os.chdir("/tritonserver/clients/bin")
        os.system(f"./perf_analyzer -i grpc -b {str(batch_size)} -m {str(index)} -u {server} -f {csv_file_path}")
        if os.path.exists(csv_file_path) : 
            print("created")
        else:
            exit()
        
        data = pd.read_csv(csv_file_path)
        total_latency = pd.DataFrame(data, columns=["p50 latency"]) 
        communication = pd.DataFrame(data, columns=["Network+Server Send/Recv"]) 
        
        os.system(f"rm {csv_file_path}")
        print("rm the file")

        result.second_prediction = total_latency.values[0][0]/1000000 - communication.values[0][0]/1000000
        
        edge_results.append(result)
    else:
        edge_results.append(result)
        
os.chdir(current_path)
save_data(edge_results, "edge", args.server, application)

# Triton benchmark Scirpt End
