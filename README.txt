
--- Neural Networks prject ----------------------------------
Topic: Sequence-to-sequence in Neural Machine Translation
Authors: Eleonora Vitanza & Nicholas Redi
-------------------------------------------------------------

We use this Google Drive folder to run the project experiments on Google Colab.
It contains some subfolders and the following files:
- `main.py´
- `translator.zip´
- `slides.pdf´

------------------------------------------------------------------------------------------

The `main.py´ file is the only source code to activate all the different execution modalities.
These modalities must be specified by arguments on the command line:

- to train a model:
$ python3 main.py train

- to evaluate a model:
$ python3 main.py eval

- to use a model as a translator:
$ python3 main.py on_the_fly

In order to customize the training, there are many other optional arguments that you can specify:

- to visualize them:
$ python3 main.py -h

In `eval´ and `on_the_fly´ modes, the default model is the best one founded.
If you want to use another model, you can just add an argument like:
--model <model_name>

If you want to run the program on Google Colab, you can add:
--run_on_colab True

------------------------------------------------------------------------------------------

About subfolders:

- dataset:
    It contains the text file downloaded from the suggested website on the project proposal.
    During the training some subfolders will be created.
    These will contain the splits `train_set.txt´, `val_set.txt´ and `test_set.txt´, according to the choice of the argument:
    --frac <number_for_which_divide>
    It will eventually reduce the dataset (e.g. --frac 3 --> use the 33.3% of the entire dataset)

- vocabularies:
    It contains the text files with the english and italian vocabularies, created during the last training (they will be overwritten every time)

- wembedds:
    It contains two different text files with the pretrained embedding values for English.
    You can use them adding the argument:
    --embed_file <embed_file_name>

- models:
    It contains all the models we trained during the experimentation.
    File names are automatically inferred according to the specified parameters.

- plots:
    It contains the plots of the loss for each training.
    File names are automatically inferred according to the specified parameters.

- tests:
    It contains the text files of the translations on the entire test set for each training and the associated BLEU score.
    File names are automatically inferred according to the specified parameters.

- att_plots:
    It contains the attention weights plots in the case of attention-based models.
    File names are automatically inferred according to the specified parameters.
    Notice that this folder is used just for on_the_fly translations on Google Colab.
    The attached zip does not contain it because in this case we use a graphical interface.

------------------------------------------------------------------------------------------

The `translator.zip´ file has a similar structure of the Drive folder but with just one model (the best one founded).
You can download it to try the translator in a local PC through a graphical interface.
To do this, you need to install the following dependencies:
- numpy==1.18.5
- torch==1.6.0
- torchtext==0.7.0
- matplotlib==3.2.2
- PyQt5==5.14

To test the translator just run:
$ python3 main.py on_the_fly
