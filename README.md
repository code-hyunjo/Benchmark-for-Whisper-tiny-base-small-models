# Benchmark for Whisper tiny, base, small models

## Introduction
The proportion of Korea's elderly population is steadily increasing, and it is expected that Korea will enter a super-aging society by 2025. In line with these social changes, people are interested in AI care for the elderly. However, a shortage of skilled AI experts and a lack of high-quality medical/healthcare data hinder the development of medical AI. Also, there are difficulties in accessing medical data due to privacy and security issues. To overcome these limitations, I am researching a methodology for learning models without medical data. Herein, this code presents a code that creates a benchmark that can identify the performance of Whisper which is a general-purpose speech recognition model and provide evaluation standards for subsequent tasks.

## About whisper tiny, base, small
There are five different model sizes available, four of which are English-only versions, designed to balance performance in terms of speed and accuracy. Listed below are the names of these models, along with their estimated memory requirements and relative inference speeds compared to the largest model; however, actual performance may vary based on various factors including the hardware in use.
![image](https://github.com/code-hyunjo/Benchmark-for-Whisper-tiny-base-small-models/assets/173684746/34f82a4f-f2e9-43b4-8797-d91c92263f5f)
More specific information can be obtained from the URL below.
https://github.com/openai/whisper

## Data
### Data information
To evaluate Whisper's performance on Korean medical data, medical data was downloaded from the AI ​​hub. You can download it from the URL below.
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=208

The total number of usable data is 216,581, and the json file contains labeltext, gender, dialect, age, region, sampling rate, qualitystatus, etc.

### Data extraction
To reflect the properties of the entire data, training, validation, and test data were extracted equal to the ratio of dialect, gender, and age of the entire data. It was executed with "medical data_preprocessing.ipynb" and "medical data_data extracting.ipynb"

## Setup
To use Whisper and "Whisper benchmarking.ipynb" smoothly, you must first run the code below.
```
!pip install git+https://github.com/openai/whisper.git
!pip install --upgrade librosa jiwer evaluate
```
After that, install whisper_train and other necessary libraries and run.
```
import train_whisper

import torch
import pandas as pd
import time, json
```

## Hyperparameter
The hyperparameters below were used to obtain benchmarking under the same conditions.
```
'batch_size' : 4,
'epochs' : 20,
'lr' : 2e-5,
'max_len' : 80,
'loss_function' = torch.nn.CrossEntropyLoss
'optimizer' = AdamW
'metric' = CER
```

## Results
Benchmarking could be obtained by running the code below.
```
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

f = open('put your json file code', 'r')
logging = json.load(f)
f.close()

res = []
for ver in logging['result_ver'].keys():
    for line in logging['result_ver'][ver]['result']:
        res.append([ver] + line)

df = pd.DataFrame(res, columns=['ver', 'epoch', 'train_loss', 'train_cer', 'valid_loss', 'valid_cer'])
melted_df = pd.melt(df, id_vars=['ver', 'epoch'], value_vars=['train_loss', 'train_cer', 'valid_loss', 'valid_cer'],
                    var_name='metric', value_name='value')
melted_df['data'] = melted_df['metric'].apply(lambda x: 'train' if 'train' in x else 'valid')
melted_df['metric'] = melted_df['metric'].apply(lambda x: 'loss' if 'loss' in x else 'cer')

import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(data=result_df, x='epoch', y='loss', kind = 'line', hue = 'data')
plt.show()
sns.relplot(data=result_df, x='epoch', y='cer', kind = 'line', hue = 'data')
plt.show()

```
### Whisper tiny
![image](https://github.com/code-hyunjo/Benchmark-for-Whisper-tiny-base-small-models/assets/173684746/84977618-b853-4d57-816d-fd450e3fd761)
![image](https://github.com/code-hyunjo/Benchmark-for-Whisper-tiny-base-small-models/assets/173684746/d29bf264-5087-49a4-b443-3e8c2b9129cd)
### Whisper base
![image](https://github.com/code-hyunjo/Benchmark-for-Whisper-tiny-base-small-models/assets/173684746/ef8c90a4-fb77-4dbf-b615-82c837a30179)
![image](https://github.com/code-hyunjo/Benchmark-for-Whisper-tiny-base-small-models/assets/173684746/362e0d58-b6bd-4fe6-8e1b-7623a9f46a6d)
### Whisper small
![image](https://github.com/code-hyunjo/Benchmark-for-Whisper-tiny-base-small-models/assets/173684746/01ecfd31-49aa-4387-81a4-d29c03ec8826)
![image](https://github.com/code-hyunjo/Benchmark-for-Whisper-tiny-base-small-models/assets/173684746/03071add-1048-4f5f-bbb9-bd54d0ffe890)

The above experiment results can be used to evaluate performance in other tasks.
