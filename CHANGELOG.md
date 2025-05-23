## 1dc7ad9
 #### 2025-05-09 

minor changes



## 22c5881
 #### 2025-05-09 

Added detailed documentation to model module. Also, added the 
un_cotrain() function to the bottom of model.py.



## 262ecb0
 #### 2025-05-09 

Added partial training logs from running the original KGPL Tensorflow code on the music dataset.



## 154bad4
 #### 2025-05-09 

Updated requirements. The requirements.txt should be up to date, based on the imports needed to run the notebooks/scripts.



## 30d2b59
 #### 2025-05-09 

Added docstrings for the grouper function. All docstrings for the 'utils/functions' module are complete.



## 9efc2e5
 #### 2025-05-09 

Added final version of KGPL experiment on the Lastfm (music) dataset.



## 0cc0de2
 #### 2025-05-08 

Updated the changelog.



## d332a34
 #### 2025-05-08 

Added cold_start_eval function. The function evaluates the recommendation model on cold start users. Detailed docstrings are included.



## 7084d40
 #### 2025-05-08 

Simplified data classes. Removed the _fix_set_for_positives function and updated documentation accordingly.



## 21f1d64
 #### 2025-05-08 

Finished docstrings for the evaluation module. See the eval.py script for detailed documentation on running Top-K evaluation for the recommender model.



## 5cb01ce
 #### 2025-05-08 

Edited docstrings for sample_positive method in KGPLDataset parent class.



## d7f7fac
 #### 2025-05-08 

Added docstrings for data_classes module. All classes and class methods in the data_classes module have detailed docstrings and notes.



## 56f57a1
 #### 2025-05-07 

Added detailed docstrings for all functions in the preprocess script.



## e71252b
 #### 2025-05-07 

Reran train and evaluation in notebook. The new preprocessing scripts should be more reproducible.



## 0b5c564
 #### 2025-05-07 

Added descriptive docstrings and type hints to all functions in make_path_list script.



## 08f453e
 #### 2025-05-07 

Updated changelog with all recent commits.



## 4280815
 #### 2025-05-07 

Consistency fix in main function. The get_paths function does not take an rng generator.



## 8f49990
 #### 2025-05-07 

Testing update_changelog script.



## 26ceac0
 #### 2025-05-07 

Created script to simplify updating the changelog.



## 5268e00
 #### 2025-05-07 

Updated changelog.



## 85db2e5
 #### 2025-05-07 

Quick fix to make_path_list.py. Added the 
ng generator to construct the adjacency matrix.



## 2d291a7
 #### 2025-05-07 

Created a Changelog. Instructions to add to the Changelog are in the README.



## 4a02fe9
 #### 2025-05-07 

Added reproducibility measures to make_path_list.py: Constructed and passed independent random generators per parallel process, sorted the set of neighbors so the BFS traversal order is fixed and the resulting set of paths should be consistent across runs.



## 03f0729
 #### 2025-05-07 

Changed first line to clone into test branch instead of dev branch



## fd48db2
 #### 2025-05-07 

Added instructions for adding to CHANGELOG.md



## 52de67e
 #### 2025-05-07 

Added a gitignore file.



## d61d886
 #### 2025-05-05 

Add files via upload



## 4e9c2ab
 #### 2025-05-05 

Update imports



## 47c174f
 #### 2025-05-05 

Update eval.py



## 89b6760
 #### 2025-05-05 

Update model.py



## 70a3ceb
 #### 2025-05-05 

Update model.py



## 826cca0
 #### 2025-05-05 

User and item embedding size



## 401262e
 #### 2025-05-05 

Update data_classes.py



## 5f969d8
 #### 2025-05-05 

Unindent



## d52ad1a
 #### 2025-05-05 

Update data_classes.py



## 89d8f3e
 #### 2025-05-05 

Update data_classes.py



## 4b57581
 #### 2025-05-05 

KGPLCOT init takes exp object



## 18eb2f6
 #### 2025-05-05 

update config keys



## 1c71f15
 #### 2025-05-05 

Update paths to match config



## 3f32383
 #### 2025-05-05 

Update music.yaml



## ce6b6dc
 #### 2025-05-05 

Update base path



## 3f1db3f
 #### 2025-05-05 

relative import



## 0ab7b69
 #### 2025-05-05 

Update functions.py



## 4fc79a8
 #### 2025-05-05 

Fix indent and imports



## b8298b3
 #### 2025-05-05 

Update data_classes.py



## d4e10f5
 #### 2025-05-04 

Delete conf/movie.yaml



## dd46c02
 #### 2025-05-04 

Delete conf/music.yaml



## e34e1a1
 #### 2025-05-04 

Create music.yaml



## 97e61e8
 #### 2025-05-04 

Create movie.yaml



## 88b4a14
 #### 2025-05-04 

Create preprocess.yaml



## e945c28
 #### 2025-05-04 

Update data_classes.py



## 93cfd9f
 #### 2025-05-04 

imports



## c1ad448
 #### 2025-05-04 

Delete data/movie/_



## edd7e74
 #### 2025-05-04 

Add files via upload



## 0c3b076
 #### 2025-05-04 

Delete data/music/_



## d768763
 #### 2025-05-04 

Add files via upload



## 7691451
 #### 2025-05-04 

Create _



## 68e94da
 #### 2025-05-04 

Delete data/music



## 4a67cd0
 #### 2025-05-04 

Create _



## 7fabff5
 #### 2025-05-04 

Create music



## af80fe1
 #### 2025-05-04 

Update data_classes.py



## 47d8253
 #### 2025-05-04 

add dataloaders



## 5644517
 #### 2025-05-04 

Create requirements.txt



## 71ea1bf
 #### 2025-05-04 

Create eval.py



## 3d003be
 #### 2025-05-04 

Create model.py



## 4d0dc36
 #### 2025-05-04 

Create movie.yaml



## 70e9a8b
 #### 2025-05-04 

Create music.yaml



## 2d77046
 #### 2025-05-04 

Update data_classes.py



## e5e6ad9
 #### 2025-05-04 

Create data_classes.py



## bb72fdc
 #### 2025-05-04 

Create functions.py



## 5fecf9b
 #### 2025-05-04 

Delete Data directory



## 9bf6e43
 #### 2025-05-04 

Update and rename kg (1).txt to kg.txt



## 33fa20e
 #### 2025-05-04 

Update and rename item_index2entity_id (2).txt to item_index2entity_id.txt



## 875125e
 #### 2025-05-04 

Delete Data/movie/_



## 9d3ef3c
 #### 2025-05-04 

Add files via upload



## cdd77a8
 #### 2025-05-04 

Create _



## 440b6d0
 #### 2025-05-04 

Delete Data/music/_



## a9b56ce
 #### 2025-05-04 

Update and rename item_index2entity_id (1).txt to item_index2entity_id.txt



## 153593f
 #### 2025-05-04 

Add files via upload



## 7ccef83
 #### 2025-05-04 

Create _



## f0a3cfb
 #### 2025-05-04 

Delete Data/book/_



## 9ffb731
 #### 2025-05-04 

Add files via upload



## 8cb7929
 #### 2025-05-04 

Create _



## 742c307
 #### 2025-05-04 

Create preprocess.py



## dd7974d
 #### 2025-05-04 

Create make_path_list.py



## 20dd465
 #### 2025-05-04 

Delete models directory



## eb23f92
 #### 2025-05-04 

Delete FINAL_PROJECT.ipynb



## a79cb58
 #### 2025-05-04 

Delete Pipeline_working_except_positive_sampling.ipynb



## af3ee59
 #### 2025-05-04 

Delete .gitignore



## 53a0db0
 #### 2025-05-04 

Delete FINAL_PROJECT (1).ipynb



## c42985a
 #### 2025-05-04 

Delete environment.yml



## 8cd8274
 #### 2025-04-27 

Merge pull request #1 from dna-witch/sm-dev

Sm dev

## 2a87d73
 #### 2025-04-27 

Update README.md



## 9798199
 #### 2025-04-27 

Update README.md



## 0f11c23
 #### 2025-04-26 

Add files via upload



## 62aa725
 #### 2025-04-26 

Merge branch 'sm-dev' of https://github.com/dna-witch/KGPL-PyTorch into sm-dev



## 90d2e11
 #### 2025-04-26 

minor changes



## 1026651
 #### 2025-04-26 

Add files via upload



## 63a47a6
 #### 2025-04-26 

added first draft of training function to utils



## ed9f418
 #### 2025-04-26 

Slightly cleaned up refactored versions of utils.py and kgpl.py. Needs more testing!



## 6560022
 #### 2025-04-26 

Refactored versions of kgpl.py and utils.py. Still need to test all scripts!



## a6e166f
 #### 2025-04-26 

Refactored aggregators module to be more Pythonic and started the cotraining modules. Also created an Embedding Matrix class object in utils.py



## 89dfb61
 #### 2025-04-26 

Converted Aggregators module to PyTorch



## 94923f5
 #### 2025-04-24 

First stages of preprocessing



## 0222642
 #### 2025-04-20 

Initial commit


