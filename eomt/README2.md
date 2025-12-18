# Anomaly Segmentation Course Project (EoMT)

This project implements **Entropic Open-Set Mutual Training (EoMT)** for semantic segmentation and anomaly detection. It evaluates how well the model can detect unknown objects (anomalies) in road scenes while maintaining high accuracy on standard classes.

## 📂 Key Files & Description

Here is what each important file in this repository does:

### 1\. Core Scripts

  * **`solve_miou.py`**
      * **What it does:** Calculates the standard accuracy (**mIoU**) of the model on the Cityscapes validation set. This proves the model knows "what is a car" and "what is a road." but I dont file eomt_weight file so the code does not work
  * **`solve_step5_smiyc_ro.py`**
      * **What it does:** Runs the main anomaly detection benchmarks (MSP, MaxLogit, RbA) on the **RoadAnomaly** and **SMIYC** datasets. Generates the numbers for Table 1. but the smiyc dataset is not available I can run this code sucessfully
  * **`solve_step5_temperature.py`**
      * **What it does:** Experiments with **Temperature Scaling** (T=0.5, T=1.0, etc.) to see if changing model confidence improves anomaly detection. Generates the numbers for Table 2.
  * **`inference.ipynb`**
      * **What it does:** A Jupyter Notebook to visualize predictions. It takes a single image and shows the segmentation mask and the anomaly heatmap side-by-side.

-----

## 🚀 How to Run

### 1\. Setup

Ensure your data is organized as follows:

```
project_folder/
├── data/
│   ├── leftImg8bit_trainvaltest/  (Cityscapes Images)
│   ├── gtFine_trainvaltest/       (Cityscapes Labels)
│   ├── RoadAnomaly_jpg/           (Anomaly Dataset 1)
│   └── SMIYC_RO21/not complete     (Anomaly Dataset 2)
├── eomt_pretrained.pth            (Model Weights)
└── solve_miou.py                  (Scripts)
```



### 2\. Run Files

```bash
python [name]].py --weights eomt_pretrained.pth
```

-----

