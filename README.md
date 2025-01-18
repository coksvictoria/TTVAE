# TTVAE: Transformer-based Generative Modeling for Tabular Data Generation
#### Alex X. Wang and Binh P. Nguyen(https://people.wgtn.ac.nz/b.nguyen) ∗

## Abstract
Tabular data synthesis presents unique challenges, with Transformer models remaining underexplored despite the applications of Variational Autoencoders and Generative Adversarial Networks. To address this gap, we propose the Transformer-based Tabular Variational AutoEncoder (TTVAE), leveraging the attention mechanism for capturing complex data distributions. The inclusion of the attention mechanism enables our model to understand complex relationships among heterogeneous features, a task often difficult for traditional methods. TTVAE facilitates the integration of interpolation within the latent space during the data generation process. Specifically, TTVAE is trained once, establishing a low-dimensional representation of real data, and then various latent interpolation methods can efficiently generate synthetic latent points. Through extensive experiments on diverse datasets, TTVAE consistently achieves state-of-the-art performance, highlighting its adaptability across different feature types and data sizes. This innovative approach, empowered by the attention mechanism and the integration of interpolation, addresses the complex challenges of tabular data synthesis, establishing TTVAE as a powerful solution.

## Illustration of TTVAE with latent space interpolation
![alt text](ttvaelsi.svg)

### A detailed demo can be found in Demo_TTVAE.ipynb.

### Installing Dependencies

Python version: 3.10

```
pip install -r requirements.txt
```
### Preparing Datasets

#### Save dataset properties into JSON file in the following folder.
\data_profile\{dataname.json}

####Download raw dataset:

```
python s1_download_dataset.py
```

####Process dataset:

```
python s2_process_dataset.py
```

### Training Models

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
