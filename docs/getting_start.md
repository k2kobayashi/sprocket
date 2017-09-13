# Getting start with VC example
After installation finished, you can easily try voice conversion (VC) example scripts in "/sprocket/example"

```
cd example
```

## Download dataset from Voice Conversion Challenge 2016 (VCC2016)
In this example, you first prepare parallel speech dataset of source and target speaker in the "data/wav" directory.
In this tutorial, you can download VCC2016 speech database by executing following command.

```
python download_vcc2016dataset.py
```

Now, you can find .wav files in "data/wav" in each speaker.

## Initialization
For the initializations, you need to perform following 3 steps.

1. Prepare lists for training and evaluation in each source and target speaker
2. Prepare configure files of the source and target speaker and speaker-pair
3. Modify the range for F0 extraction 

### 1. Prepare training and evaluation lists
To create lists of source speaker (e.g., "SM1") and target speaker (e.g., "TF1"), run following command in example directory

```
python initialize.py -1 SM1 TF1 16000
```
where 16000 is a sampling rate of .wav file.
"-1" option indicates a flag to generate training and evaluation lists of source and target speakers.

You can change the number of speech samples to be used in the training and evaluation by editing the lists in "/list".
In this example, we make lists shorter by manually modification
We change the number of training speech samples to 30 and the number of evaluation speech samples to 10 as below.

list/SM1_train.list: 

``` 
SM1/100001
SM1/100002
SM1/100003
SM1/100004
SM1/100005
SM1/100006
SM1/100007
SM1/100008
SM1/100009
SM1/100010
SM1/100011
SM1/100012
SM1/100013
SM1/100014
SM1/100015
SM1/100016
SM1/100017
SM1/100018
SM1/100019
SM1/100020
SM1/100021
SM1/100022
SM1/100023
SM1/100024
SM1/100025
SM1/100026
SM1/100027
SM1/100028
SM1/100029
SM1/100030
```
list/SM1_eval.list: 

```
SM1/100031
SM1/100032
SM1/100033
SM1/100034
SM1/100035
SM1/100036
SM1/100037
SM1/100038
SM1/100039
SM1/100040
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
python initialize.py -2 SM1 TF1 16000
```
where "-2" option indicates a flag to generate configure files for source and target speakers.
By executing this scripts, speaker-dependent YAML file (e.g., conf/speaker/SM1.yml) and speaker-pair dependent YAML file (e.g., conf/pair/SM1-TF1.yml) are generated. 

### 3. Modify F0 extraction range
In order to achieve better sound quality and conversion accuracy of the converted voice, it is necessary to designate appropriate parameters. 
In this step, we describe about how to define the F0 ranges.
First you run following command to generated F0 histograms.

```
python initialize.py -3 SM1 TF1 16000
```
where "-3" option indicates a flag to generate F0 histograms of the source and target speakers.
After finishing this commands, you can find the histograms in "conf/figure" directory.
Here is a example figure in "conf/figure/SM1_f0histogram.png".

![Example](./png/f0histogram_example.png)

Based on this figure, you manually modify "minf0" and "maxf0" in speaker-dependent YAML file (e.g., conf/speaker/TF1.yml) in each speaker.

## Run VC
Now you can perform VC using "run_sprocket.py"

```
python run_sprocket.py -1 -2 -3 -4 -5 SM1 TF1
```
This command perform based on following figure step by step.

![VCflow](./png/vc_flow.png)


## Run DIFFVC
If you want to perform DIFFVC with F0 transformation, you need to perform F0 transformation of speech samples of source speaker.
To perform F0 transformation, you run following command

```
python run_f0_transformation.py SM1 TF1
```
After command finished, you can find F0 transformed wav files in "data/wav" directory.
Using this wav file as a new source speaker, you can perform DIFFVC with F0 transformation.
Note that you need to perform initialization and run VC steps for F0 transformed source speaker and target speaker again.
