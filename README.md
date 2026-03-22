# GTR+: Generative Retrieval for Unsupervised Text-Based Person Search

This is the official PyTorch implementation of our paper **Generative Retrieval for Unsupervised Text-Based Person Search**. The paper link will be released soon.

## Highlights

We propose **GTR+** for unsupervised text-based person search, removing the need for expensive human-annotated descriptions. GTR+ combines:

- a **three-tier description generation framework** for producing fine-grained and diverse pseudo texts;
- an **adaptive confidence-weighted retrieval learning framework** to alleviate noisy supervision;

- **LargeFine-Person**, a large-scale benchmark for unsupervised TBPS pre-training.

![The structure of GTR+ model](figs/model-structure.png)

## Updates

- [2026-3-20] Initial release of code.
- ...

## Requirements

Our experiments are mainly conducted on NVIDIA L40 GPUs. The code should also run on other GPUs with sufficient memory.

More dependency details are provided in [requirements.txt](requirements.txt).

## Quick Start

```bash
git clone ...
cd ...
conda create -n blip -y python=3.10
conda activate blip
pip install -r requirements.txt
```

## Training/Evaluation

The following scripts provide an example for training and evaluation.
Please modify the dataset paths and checkpoint paths in the scripts before running.

```bash
# Training
bash shell/train.sh

# Evaluation
bash shell/eval.sh
```



## Prepare Datasets

Download the [**CUHK-PEDES**](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) dataset, [**ICFG-PEDES**](https://github.com/zifyloo/SSAN) dataset and [**RSTPReid**](https://github.com/NjtechCVLab/RSTPReid-Dataset) dataset.

```
dataset_root/
├── CUHK-PEDES/
│   ├── imgs/
│   │   ├── cam_a/
│   │   ├── cam_b/
│   │   └── ...
│   └── reid_raw.json
├── ICFG-PEDES/
│   ├── imgs/
│   │   ├── test/
│   │   └── train/
│   └── ICFG_PEDES.json
├── RSTPReid/
│   ├── imgs/
│   └── data_captions.json
└── LargeFine-Person/
    ├── imgs/
    ├── LargeFine_Person_qa.json
    ├── LargeFine_Person_com.json
    └── LargeFine_Person_sty.json
```

### LargeFine-Person Dataset

Download our pre-training dataset  [**LargeFine-Person**](https://drive.google.com/drive/folders/1tfJwTlLawZDEcxAhrCubpkjzApRQIdvH?usp=drive_link)

![Samples of our LargeFine-Person Dataset](figs/LargeFine-samples.png)![Samples of our LargeFine-Person Dataset](figs/LargeFine-samples.png)



## Unsupervised TBPS Results with [BLIP](https://github.com/salesforce/BLIP) as Baseline

**CUHK-PEDES**

| Method                                                       |                  Baseline                  | Fine-tuning |    R@1    | R@5       | R@10      | mAP       |                          Checkpoint                          |
| ------------------------------------------------------------ | :----------------------------------------: | :---------: | :-------: | --------- | --------- | --------- | :----------------------------------------------------------: |
| [GTR](https://arxiv.org/abs/2305.12964)                      | [BLIP](https://github.com/salesforce/BLIP) |             |   47.53   | 68.23     | 75.91     | 42.91     |                              /                               |
| [GAAP](https://www.ijcai.org/proceedings/2024/116)           | [BLIP](https://github.com/salesforce/BLIP) |             |   47.64   | 67.79     | 76.08     | 41.28     |                              /                               |
| [MUMA](https://ojs.aaai.org/index.php/AAAI/article/view/32543 ) | [BLIP](https://github.com/salesforce/BLIP) |             |   59.52   | 77.79     | 84.65     | 52.75     |                              /                               |
| **GTR+**                                                     | [BLIP](https://github.com/salesforce/BLIP) |             | **61.35** | **79.35** | **85.75** | **55.75** | **[Download](https://drive.google.com/file/d/1ZuFGroJ-Iqx30i73LeOSI_B-8Jcc5qcn/view?usp=drive_link)** |
| **GTR+ (Pre-trained)**                                       | [BLIP](https://github.com/salesforce/BLIP) |      ✗      | **62.65** | **78.80** | **84.76** | **55.27** | **[Download](https://drive.google.com/file/d/1ZDo7KgSwVDjFNTVuyWqfMZ3awPuYUqhg/view?usp=drive_link)** |
| **GTR+ (Pre-trained)**                                       | [BLIP](https://github.com/salesforce/BLIP) |      ✓      | **64.65** | **80.72** | **86.78** | **58.67** | **[Download](https://drive.google.com/file/d/1oHhPkk52BCljjVhLzHti-HwDin3HYGtk/view?usp=drive_link)** |

**ICFG-PEDES**

| Method                                                       |                  Baseline                  | Fine-tuning |    R@1    | R@5       | R@10      | mAP       |                          Checkpoint                          |
| ------------------------------------------------------------ | :----------------------------------------: | :---------: | :-------: | --------- | --------- | --------- | :----------------------------------------------------------: |
| [GTR](https://arxiv.org/abs/2305.12964)                      | [BLIP](https://github.com/salesforce/BLIP) |             |   28.25   | 45.21     | 53.51     | 13.82     |                              /                               |
| [GAAP](https://www.ijcai.org/proceedings/2024/116)           | [BLIP](https://github.com/salesforce/BLIP) |             |   27.12   | 44.91     | 53.56     | 11.43     |                              /                               |
| [MUMA](https://ojs.aaai.org/index.php/AAAI/article/view/32543 ) | [BLIP](https://github.com/salesforce/BLIP) |             |   38.11   | 56.01     | 63.96     | 19.02     |                              /                               |
| **GTR+**                                                     | [BLIP](https://github.com/salesforce/BLIP) |             | **47.81** | **64.97** | **71.94** | **28.75** | **[Download](https://drive.google.com/file/d/1LPTIfh6FbrFh_sBhoLiaDD5Q0UvtTZnG/view?usp=drive_link)** |
| **GTR+ (Pre-trained)**                                       | [BLIP](https://github.com/salesforce/BLIP) |      ✗      | **47.53** | **64.32** | **71.39** | **25.38** | **[Download](https://drive.google.com/file/d/1R91hfkyvWuYPtuUv8nta5SEvRDTRdtVB/view?usp=drive_link)** |
| **GTR+ (Pre-trained)**                                       | [BLIP](https://github.com/salesforce/BLIP) |      ✓      | **52.78** | **67.94** | **73.91** | **33.99** | **[Download](https://drive.google.com/file/d/1nYTpZgZFw8AVQbYYd9_k3hgoixb1T5_q/view?usp=drive_link)** |

**RSTPReid**

|                            Method                            |                  Baseline                  | Fine-tuning |       R@1 |       R@5 |      R@10 | mAP       |                          Checkpoint                          |
| :----------------------------------------------------------: | :----------------------------------------: | :---------: | --------: | --------: | --------: | --------- | :----------------------------------------------------------: |
|           [GTR](https://arxiv.org/abs/2305.12964)            | [BLIP](https://github.com/salesforce/BLIP) |             |     45.60 |     70.35 |     79.95 | 33.30     |                              /                               |
|      [GAAP](https://www.ijcai.org/proceedings/2024/116)      | [BLIP](https://github.com/salesforce/BLIP) |             |     44.45 |     65.15 |     75.30 | 31.21     |                              /                               |
| [MUMA](https://ojs.aaai.org/index.php/AAAI/article/view/32543 ) | [BLIP](https://github.com/salesforce/BLIP) |             |     54.35 |     76.05 |     83.65 | 40.50     |                              /                               |
|                           **GTR+**                           | [BLIP](https://github.com/salesforce/BLIP) |             | **54.75** | **75.15** | **83.50** | **43.79** | **[Download](https://drive.google.com/file/d/1Dz_9rLgGPeCP4yLYH-7fFF2dYiBX3Z-W/view?usp=drive_link)** |
|                    **GTR+ (Pre-trained)**                    | [BLIP](https://github.com/salesforce/BLIP) |      ✗      | **52.00** | **74.05** | **82.35** | **38.72** | **[Download](https://drive.google.com/file/d/11A8LlAsA_2kFJK5bcJS5HnF8BrDVpXr8/view?usp=drive_link)**                     |
|                    **GTR+ (Pre-trained)**                    | [BLIP](https://github.com/salesforce/BLIP) |      ✓      | **55.70** | **76.55** | **84.25** | **43.86** | **[Download](https://drive.google.com/file/d/1R5Y9P5-KJJtr393q6kGcQIU5EIo6ZXmi/view?usp=drive_link)** |



## Supervised TBPS Results with [IRRA](https://github.com/anosorae/IRRA/tree/main) as Baseline

**CUHK-PEDES**

| Method   |                      Baseline                      | Fine-tuning | R@1   | R@5   | R@10  | mAP   |                          Checkpoint                          |
| -------- | :------------------------------------------------: | :---------: | ----- | ----- | ----- | ----- | :----------------------------------------------------------: |
| **GTR+** | [IRRA](https://github.com/anosorae/IRRA/tree/main) |             | 59.44 | 78.54 | 85.22 | 54.11 | **[Download](https://drive.google.com/file/d/1hbLyoTAeA8HvA5zPM51_dNgQKAUOq5pc/view?usp=drive_link)** |
| **GTR+** | [IRRA](https://github.com/anosorae/IRRA/tree/main) |      ✓      | 77.13 | 90.82 | 94.49 | 68.37 | **[Download](https://drive.google.com/file/d/1xss3Wu2NVYm0vP9WmMzREIHSDWaLAVxL/view?usp=drive_link)** |

**ICFG-PEDES**

| Method   |                      Baseline                      | Fine-tuning | R@1   | R@5   | R@10  | mAP   |                          Checkpoint                          |
| -------- | :------------------------------------------------: | :---------: | ----- | ----- | ----- | ----- | :----------------------------------------------------------: |
| **GTR+** | [IRRA](https://github.com/anosorae/IRRA/tree/main) |             | 43.77 | 60.77 | 68.05 | 22.30 | **[Download](https://drive.google.com/file/d/1jvL9Q7hjn0W_0JrFPqvnHVdYImUphOMh/view?usp=drive_link)** |
| **GTR+** | [IRRA](https://github.com/anosorae/IRRA/tree/main) |      ✓      | 67.80 | 82.81 | 87.66 | 41.00 | **[Download](https://drive.google.com/file/d/1E9s7-Gd-d8Krgo_1WmCapYSml_nwandd/view?usp=drive_link)** |

**RSTPReid**

| Method   |                      Baseline                      | Fine-tuning | R@1   | R@5   | R@10  | mAP   |                          Checkpoint                          |
| -------- | :------------------------------------------------: | :---------: | ----- | ----- | ----- | ----- | :----------------------------------------------------------: |
| **GTR+** | [IRRA](https://github.com/anosorae/IRRA/tree/main) |             | 50.45 | 73.45 | 82.35 | 37.68 | **[Download](https://drive.google.com/file/d/1WJhGlMqsJEDqN0UyVX5AxtRaZjev_V2_/view?usp=drive_link)** |
| **GTR+** | [IRRA](https://github.com/anosorae/IRRA/tree/main) |      ✓      | 69.05 | 86.90 | 92.25 | 54.19 | **[Download](https://drive.google.com/file/d/1sZ7670RJMjOx-Xs7l3UHZGsVEzhA4EHW/view?usp=drive_link)** |



## More Examples

More qualitative examples of generated descriptions and retrieval results are shown below.

![More Examples](figs/examples.png)



## Citation

If you find this code useful for your research, please cite our paper.

```bibtex
coming soon
```

