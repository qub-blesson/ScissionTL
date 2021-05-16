# ScissionTL: A benchmarking tool for Accelerating Distributed Deep Neural Networks Using Transfer Layer in Edge Computing

## About the software
### ScissionTL benchmarking (For TensorFlow backend Edge)
#### Dependencies 
* Tensorflow 2.3
* NumPy
* Pillow
* matplotlib 

#### benchmarking
```python3 scissionTL_benchmark.py device tx2_gpu``` 

```python3 scissionTL_benchmark.py device tx2_cpu -dc```

You need to specify your envrionment(device or edge) to first argument.
Second argument is its name.
An output file "device-tx2_gpu.dat" will be saved which contains the benchmark data for each benchmarked model.
To disable the GPU, set the -dc to True. 

Default benchmarking models are listed below: 
```
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
```
It is downloaded from tensorflow.keras.applications  

### ScissionTL prediction(For tensorflow backend Edge)
#### Dependencies
* NumPy
* Matplotlib
#### Network Statistics
Predictions use externally provided network statistics in a csv file formatted as below:

source | destination | ping(ms) | bandwidth(mbps) 
--------|-------------|----------|-----------------
device1 | edge1 | 66.7 | 100.6
device2 | edge2 | 0.8 | 80.6
device1 | edge2 | 0.8 | 80.6


#### Querying - Criteria Format
* A number to indicate that the layer with the corresponding number must be executed on that platform.
* A number preceded by a "*" to indicate that the layer with corresponding number must not be executed on that platform.
* A string to indicate that only the system with the specified name is used for that platform.
* A string predceded by a "*" to indicate that the specified system must not be used. 

Criteria | Explanation 
-------- | ---------- 
-d “3,*10,device1” | Layer 3 must be executed on device, layer 10 must not. The device with name "device1" must be used.
-e “edge1,*edge2” | The edge2 must not be used, The edge1 must be used. 


#### Prediction
```python scissionTL_predict.py benchmark_data network_stats.csv model_to_predict -d "device_criteria" -e "edge_criteria" ```


The specified benchmark_data folder must contain data files produced by "scissionTL_benchmark" and a network statistics file which contains network connection information for each of the systems benchmarked. An example folder containing benchmark files and network statistic files has been provided ("benchmark_data").

The fastest configurations that fit the specified criteria will be printed, additionally, a graph of the fastest configuration showing per layer latencies and network + tranfer layer  overheads will be saved to the working directory.

The configurations are printed in the format:

```End-to-End latency - Total bandwidth used across the edge pipeline - Layer distribution```

The output from the above prediction is displayed below:

```
[+] densenet169 results
[1] 0.1261s - 0.1588MB - Device(tx2_gpu) = 0 - 1, Edge(ampere) = 2 - 596
[2] 0.1870s - 0.4015MB - Device(tx2_gpu) = 0 - 51, Edge(ampere) = 52 - 596
[3] 0.2068s - 0.2008MB - Device(tx2_gpu) = 0 - 139, Edge(ampere) = 140 - 596
[4] 0.2152s - 0.8029MB - Device(tx2_gpu) = 0 - 2, Edge(ampere) = 3 - 596
[5] 0.2161s - 0.8029MB - Device(tx2_gpu) = 0 - 3, Edge(ampere) = 4 - 596
[+] Graph created: densenet169-0.13s```
```

Set -versus if you want to see difference between TL and without TL

```python scissionTL_predict.py benchmark_data network_stats.csv model_to_predict -d "device_criteria" -e "edge_criteria" -versus```

The output from the above prediction is displayed below:

```
[+] densenet169 results
[1] tl_layer_time: 0.0008 + tl_dataconvert_time: 0.0001 + tl_transfer_time: 0.0490 vs orignal_tranfer_time: 0.1138 || tl_output_size: 0.1588MB vs orig_output_size: 0.6349MB 
 (*) total time - 0.1261s - Device(tx2_gpu) = 0 - 1, Edge(ampere) = 2 - 596
[2] tl_layer_time: 0.0009 + tl_dataconvert_time: 0.0002 + tl_transfer_time: 0.0820 vs orignal_tranfer_time: 0.2459 || tl_output_size: 0.4015MB vs orig_output_size: 1.6058MB 
 (*) total time - 0.1870s - Device(tx2_gpu) = 0 - 51, Edge(ampere) = 52 - 596
[3] tl_layer_time: 0.0008 + tl_dataconvert_time: 0.0001 + tl_transfer_time: 0.0547 vs orignal_tranfer_time: 0.1366 || tl_output_size: 0.2008MB vs orig_output_size: 0.8029MB 
 (*) total time - 0.2068s - Device(tx2_gpu) = 0 - 139, Edge(ampere) = 140 - 596
[4] tl_layer_time: 0.0009 + tl_dataconvert_time: 0.0006 + tl_transfer_time: 0.1366 vs orignal_tranfer_time: 0.4643 || tl_output_size: 0.8029MB vs orig_output_size: 3.2114MB 
 (*) total time - 0.2152s - Device(tx2_gpu) = 0 - 2, Edge(ampere) = 3 - 596
[5] tl_layer_time: 0.0009 + tl_dataconvert_time: 0.0006 + tl_transfer_time: 0.1366 vs orignal_tranfer_time: 0.4643 || tl_output_size: 0.8029MB vs orig_output_size: 3.2114MB 
 (*) total time - 0.2161s - Device(tx2_gpu) = 0 - 3, Edge(ampere) = 4 - 596
 ```
