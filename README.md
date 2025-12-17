# When Imbalance Comes Twice : Active Learningunder Simulated Class Imbalance and LabelShift in Binary Semantic Segmentation

Code used to reproduce the results of the paper. 
I want to apologize in advance for the quality of the code that is not the best for two reasons :

- for intelectual property reasons i am not able to open the full package i developed to do the active learning, so here is only the parts of le package that allow the reproduction of the results. it is not beautiful.
- I am not a developer and i try to avoid using GenAI, so i do my best to make things work.

All the metada regarding active learning are contained in the config.yaml file.
I have to say that the experiment is very computationally expensive, indeed, all the AL cycle have to be ran on each dataset with each strategies with each imbalance a certain number of reps. So i would advice you to run a low number of repetitions.


## Requirements

The requirements are contained in the files, all the source code is coded with a CLI using Typer. 
In order to replicate the environement i used to run the experiments use uv here and it should work.

```bash
uv venv --python 3.10
uv sync
```


## Step 0 : Download the datasets

The **potato disease dataset** can be downloaded here [https://universe.roboflow.com/anup-kaygm/potato_disease-binb3](https://universe.roboflow.com/anup-kaygm/potato_disease-binb3).

The **severstal dataset** can be downloaded here [https://www.kaggle.com/c/severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection).


## Step 1 : Patching the datasets

After downloading the datasets, you have to patch them in order to create imbalanced datasets by sampling images according to a certain defect proportion.

The AL operates on the created patches datasets. The AL creates CSV files containing test metrics for each experiments and a record of images selected. Thos log files were used to generate the plot present in the article.


## Step 2 : Doing the AL on Patched datasets

This is the main.py file. 
It uses acquisition functions defined in the active learning file.

If you have any issues reproducing the results, don't hesitate to contact me, i would me please to fix any issues you might have.