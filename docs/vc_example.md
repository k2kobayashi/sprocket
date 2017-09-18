# Getting start with VC example
After installation finished, you can try an example script of voice conversion (VC) in "/sprocket/example".

```
cd example
```

## Download dataset from Voice Conversion Challenge 2016 (VCC2016)
In this example, you first prepare parallel speech dataset of source and target speaker in the "data/wav" directory.
In this tutorial, you can download VCC2016 speech database by executing following command.

```
python download_vcc2016dataset.py
```

Now, you can find speech samples in "data/wav" directory in each speaker.

## Initialization
For the initialization step, you need to perform following 3 steps.

1. Prepare lists for training and evaluation in each source and target speaker
2. Prepare configure files of the source and target speakers and a speaker-pair
3. Modify the F0 extraction ranges of source and target speakers 

### 1. Prepare training and evaluation lists
To create lists of the source speaker (e.g., "SF1") and target speaker (e.g., "TF1"), run following command in example directory

```
python initialize.py -1 SF1 TF1 16000
```
where "16000" is the sampling rate of .wav file.
"-1" option indicates a flag to generate training and evaluation lists of source and target speakers.

You can change the number of speech samples to be used in the training and evaluation process by editing the lists in "/list" directory.
In this example, we make these lists shorter by manually modification.
We change the number of training speech samples to 30 and the number of evaluation speech samples to 10 as below.

list/SF1_train.list: 

``` 
SF1/100001
SF1/100002
SF1/100003
SF1/100004
SF1/100005
SF1/100006
SF1/100007
SF1/100008
SF1/100009
SF1/100010
SF1/100011
SF1/100012
SF1/100013
SF1/100014
SF1/100015
SF1/100016
SF1/100017
SF1/100018
SF1/100019
SF1/100020
SF1/100021
SF1/100022
SF1/100023
SF1/100024
SF1/100025
SF1/100026
SF1/100027
SF1/100028
SF1/100029
SF1/100030
```
list/SF1_eval.list: 

```
SF1/100031
SF1/100032
SF1/100033
SF1/100034
SF1/100035
SF1/100036
SF1/100037
SF1/100038
SF1/100039
SF1/100040
```

list/TF1_train.list:

``` 
TF1/100001
TF1/100002
TF1/100003
TF1/100004
TF1/100005
TF1/100006
TF1/100007
TF1/100008
TF1/100009
TF1/100010
TF1/100011
TF1/100012
TF1/100013
TF1/100014
TF1/100015
TF1/100016
TF1/100017
TF1/100018
TF1/100019
TF1/100020
TF1/100021
TF1/100022
TF1/100023
TF1/100024
TF1/100025
TF1/100026
TF1/100027
TF1/100028
TF1/100029
TF1/100030
```

list/TF1_eval.list:

``` 
TF1/100031
TF1/100032
TF1/100033
TF1/100034
TF1/100035
TF1/100036
TF1/100037
TF1/100038
TF1/100039
TF1/100040
```
Note that you have to coincidence the length and order of the lists between the source and target speakers.

### 2. Generate configure files
Next, to generate configure files of speakers, run following command.

```
python initialize.py -2 SF1 TF1 16000
```
where "-2" option indicates a flag to generate configure files for the source and target speakers and the speaker-pair.
By executing this script, speaker-dependent YAML file (e.g., "conf/speaker/SF1.yml") and speaker-pair dependent YAML file (e.g., "conf/pair/SF1-TF1.yml") are generated. 
Parameters such as F0 extraction range, the number of mel-cepstrum dimension, and the number of mixture compornents are described in these YAML files, which are used for the training and conversion steps in this example. 

### 3. Modify F0 extraction range
In order to achieve better sound quality and conversion accuracy of the converted voice, it is necessary to designate appropriate parameters. 
In this step, we describe how to define the F0 ranges.
First you run following command to generated F0 histograms in each source and target speaker.

```
python initialize.py -3 SF1 TF1 16000
```
where "-3" option indicates a flag to generate the F0 histograms of the source and target speakers.
After finishing this commands, you can find the histograms in "conf/figure" directory.
Here is an example figure in "conf/figure/SF1_f0histogram.png".

![Example](png/f0histogram_example.png)

Based on this figure, you manually change the values of "minf0" and "maxf0" in speaker-dependent YAML file (e.g., "conf/speaker/TF1.yml").

## Run VC
Now you can execute VC using "run_sprocket.py"

```
python run_sprocket.py -1 -2 -3 -4 -5 SF1 TF1
```
The all procedures of "run_sprocket.py" are described in following figure.

![VCflow](png/vc_flow.png)


## Run DIFFVC
If you want to perform DIFFVC with F0 transformation, you need to perform F0 transformation of speech samples of source speaker.
To perform F0 transformation, you run following command

```
python run_f0_transformation.py SF1 TF1
```
After this command finished, you can find F0 transformed wav files in "data/wav" directory.
Using these speech samples as a new source speaker (e.g., "SF1_1.02"), you can execute DIFFVC with F0 transformation.
Note that you need to perform initialization and run VC steps for F0 transformed source speaker and target speaker again.
