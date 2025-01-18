# TTVAE: Transformer-based Generative Modeling for Tabular Data Generation

## A detailed demo can be found in Demo_TTVAE.ipynb.

## Installing Dependencies

Python version: 3.10

```
pip install -r requirements.txt
```



## Preparing Datasets

### Save dataset properties into JSON file in the following folder.
\data_profile\{dataname.json}

###Download raw dataset:

```
python s1_download_dataset.py
```

###Process dataset:

```
python s2_process_dataset.py
```

## Training Models

For non-deep models, including SMOTE and its variants, Synthpop, Copula and traditional deep generative models, including CTGAN, TVAE and CopulaGan, use the following code

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_METHODS] --mode train
```

Options of [NAME_OF_DATASET] in the sample code: adult, abalone
Options of [NAME_OF_METHODS]: smote, synthpop, copula, ctgan, tvae and copulagan

For other more advanced deep generative models, we break down the process into model training and data sampling:


```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_METHODS] --mode train
```

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_METHODS] --mode sample
```
Options of [NAME_OF_DATASET] in the sample code: adult, abalone
Options of [NAME_OF_METHODS]: ctabgan,tabddpm and ttvae


## License

This project is licensed under the Apache-2.0 License.


## Reference
We appreciate your citations if you find this repository useful to your research!
```
@article{wang2025ttvae,
  title={TTVAE: Transformer-based Generative Modeling for Tabular Data Generation},
  author={Alex X. Wang and Binh P. Nguyen},
  journal={Artificial Intelligence},
  year={2025},
  publisher={Elsevier}
}
```
