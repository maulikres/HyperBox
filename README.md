# HyperBox
This project is about our work on Hypernym Discovery. 

The task and dataset details can be found here: https://competitions.codalab.org/competitions/17119.

The code for box embedding has been adapted from: https://github.com/ralphabb/BoxE. 

## Getting started:

Clone this repo to your machine.

Run the following command:

        cd HyperBox
        
Install dependencies from *requirements.txt*.        

## Download Data:
Copy the contents of the "dataset" folder from [here](https://ubcca-my.sharepoint.com/:f:/r/personal/maulik20_student_ubc_ca/Documents/Paper/HyperBox?csf=1&web=1&e=YiqWhZ) to the *dataset* folder of the *HyperBox* repo.

Unzip both the files and rename them to "2A_med_pubmed_tokenized.txt" and "2B_music_bioreviews_tokenized.txt" respectively.

## Configuration:

Edit the configuration file *hparams.ini* to indicate if you want to train for *music* or *medical* dataset.  
You can also edit hyper-parameters for training your Gensim model and Box-Embedding model in *hparams.ini*.

## Word Embeddings:

### Pretrained:

You can download the pre-trained Gensim word embeddings from the above link. Download "gensim_model" and paste it into the HyperBox repo. Download the "word_vectors_processed" folder and paste it into the *HyperBox* repo.

### Training Gensim's Model for word embeddings:

Run the following command from the root of the HyperBox repo:

        mkdir gensim_model
        python script/preprocess/gensim_word2vec.py
        
        mkdir word_vectors_processed
        python script/preprocess/save_embeddings.py
        

## Training HyperBox Model:

After training the word embeddings model, you can train the HyperBox model by running the following command:

        python script/model/train.py
        
You can also directly download the pre-trained model from [here](https://ubcca-my.sharepoint.com/:f:/r/personal/maulik20_student_ubc_ca/Documents/Paper/HyperBox?csf=1&web=1&e=YiqWhZ).

Copy the contents of "pretrained_model" to "script/model/".

## Validation:

If you have trained from scratch, run the following command:

        python script/model/validate.py [epoch]
        
E.g If you want to predict for model saved after epoch 1200 run:

        python script/model/validate.py 1200  
        
For pre-trained models, the epoch value should be 0. 

        python script/model/validate.py 0
        
## Testing:

If you have trained from scratch, run the following command:

        python script/model/predict.py [epoch]
        
where epoch value can be found out from the best performing model obtained by *validate.py*.        
        
For pre-trained models, the epoch value should be 0.

        python script/model/predict.py 0


## SemEval2018-Task9 Metrics:

The code to find metrics is already provided by the organizers of SemEval2018-Task9.

For music corpus, Run:

        python script/model/task9-scorer.py script/model/output/music_gold.txt script/model/output/music_pred.txt

For medical corpus, Run:

        python script/model/task9-scorer.py script/model/output/medical_gold.txt script/model/output/medical_pred.txt

        
##  Cite HyperBox:

If you make use of this code, or its accompanying paper, please cite this work as follows:

Accepted at he 13th International Conference on Language Resources and Evaluation (LREC 2022).

```

To be added later on.

```

## License

Licensed under the MIT License.

Copyright (C) 2021  Maulik Parmar
