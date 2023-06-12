## Installation 

Main requirements:
```
python >= 3.8+
torch >= 1.12+
transformers >= 4.24+
accelerate >= 0.11.0
```

We recommend creating a conda environment for this project:
```
conda create --name epalm python=3.8
conda activate epalm
```
Additional dependencies can be found in `requirements.txt`.


To run video tasks, install the dependencies in [TimeSformer](https://github.com/facebookresearch/TimeSformer) (mainly fvcore and simplejson), then install it from `./TimeSformer:
```
cd TimeSformer
python setup.py build develop
```

For caption evaluation (CIDEr, BLUE ...) you need to install the following packages:
```
conda install -c bioconda perl-xml-libxml 
conda install -c conda-forge openjdk

pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"

```