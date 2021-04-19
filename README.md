

### Environment variables & dependencies

Please create a virtual environment for smoother functioning and to avoid any dependency issues. Please install Pytorch (>=0.4.1) and use the Makefile to set up the rest of the dependencies. 
```
make setup
```

### Process data
First, unpack the data files 
```
tar xvzf data-release.tgz
```
and run the following command to preprocess the datasets.
```
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```

`<dataset>` is the name of any dataset folder in the `./data` directory. In our experiments, the five datasets used are: `nell-995`, `amazon-beauty`, `amazon-cellphone`
`<gpu-ID>` is a non-negative integer number representing the GPU index.

### Train models
1. Train embedding-based models
```
./experiment-emb.sh configs/<dataset>-<emb_model>.sh --train <gpu-ID>
```

2. In the repository containing the embeddings, run the following to build the type embeddings. This will save the embeddings to the data repositories. 
```
python extract_embeddings.py -func <aggregation function> -emb_model <emb_model> -dim <dimension> -data_dir <data directory>
```
options:
-emb_model: (distmult, complex)
-fcn: (np.max, np.mean)


The following embedding-based models are implemented: `distmult`, `complex` and `conve`.

2. Train RL models 
./experiment-rs.sh configs/<dataset>-rs.sh --train <gpu-ID>
```

* Make sure you have pre-trained the embedding-based models and set the file path pointers to the pre-trained embedding-based models correctly ([example configuration file](configs/nell-995-rs.sh)).

### Evaluate pretrained models
To generate the evaluation results of a pre-trained model, simply change the `--train` flag in the commands above to `--inference`. 

```
./experiment-rs.sh configs/<dataset>-rs.sh --inference <gpu-ID>
```

Note for the NELL-995 dataset: 

  On this dataset we split the original training data into `train.triples` and `dev.triples`, and the final model to test has to be trained with these two files combined. 
  1. To obtain the correct test set results, you need to add the `--test` flag to all data pre-processing, training and inference commands.  
   
  2. Leave out the `--test` flag during development.

### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the [configuration files](configs).


### Acknowledgements

This work is built upon the [MultihopKG] (https://github.com/salesforce/MultiHopKG) work. Please refer to their implementation for further details.


