# Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph
A PyTorch implementation of KGPL. (https://arxiv.org/abs/2011.05061)



### Changelog
#### Adding to the changelog

First, identify the last commit hash recorded in `CHANGELOG.md`. Then, use the following command (replacing `LAST_COMMIT_HASH` with the actual hash):

>git log --pretty=format:"## %h%n #### %ad %n%n%s%n%n%b%n" --date=short LAST_COMMIT_HASH..HEAD >> CHANGELOG.md

This appends all new commits since `LAST_COMMIT_HASH` to the end of the changelog.

<!-- - Taylor notes 4/27 - 1030AM
  - I have a full pipeline working with datasets and dataloaders.  It trains and the loss goes down.
  - This is single learner, not colearning yet.
  - Fixed a bug where the training set would contain data without positive examples for one or more users.
  - Need to refactor dataset slightly - it's a bit hard to understand still.
  - I haven't touched the "aggregate", "get_neighbors", or aggregator objects yet, I only used the basic Aggregator that already was in the code.  Need help with this.
  - Also validation won't work yet since I need to clean up the datasets/dataloaders a bit of a refactor still.
  - Haven't gotten to evaluation yet at all either. -->
