## Format

Each line in a data file represents a knowledge graph fact. The representation format is

    entity1 \tab entity2 \tab relation

In general, each dataset folder consists of the following files:
  
    1. train.triples 
        - facts used for training
    2. dev.triples 
        - facts used for validation
    3. test.triples 
        - facts used for testing
    4. raw.kb 
        - facts in the backbone KG 
    5. raw.pgrk
        - undirected, weightless PageRank scores of nodes in the backbone KG calculated using the tool (https://github.com/timothyasp/PageRank) 
    6. raw.csv
        - An undirected representation of raw.kb (extended with reversed edges) which is used as the input file to the PageRank tool. 

* Notes: 

    1. The NELL-995 dataset contains an extra file `train.large.triples`, which consists of the facts in `raw.kb` excluding those in `dev.triples`.
    2. The NELL-995.test dataset is used to produce the NELL-995 test set results. It does not have the `train.triples file`. Instead, it contains files `train.dev.triples` which is the training file from the original data release and `train.dev.large.triples` which is the same as `raw.kb`. Both of these training files include `dev.triples` in this folder.
    3. For all datasets, 
        - train.triples/dev.triples/test.triples do not overlap with each other
        - raw.kb/dev.triples/test.triples do not overlap with each other except for NELL-995 and NELL-995.test, where raw.kb contains dev.triples.
