import nbformat

# nb_file = 'tutorial_linfa_2d.ipynb'
nb_file = 'tutorial_linfa_3d.ipynb'

with open(nb_file, "r") as file:
    nb_corrupted = nbformat.reader.read(file)

nbformat.validator.validate(nb_corrupted)
# <stdin>:1: MissingIDFieldWarning: Code cell is missing an id field, 
# this will become a hard error in future nbformat versions. 
# You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). 
# Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future.

nb_fixed = nbformat.validator.normalize(nb_corrupted)
nbformat.validator.validate(nb_fixed[1])
# Produces no warnings or errors.

with open('fixed_'+nb_file, "w") as file:
    nbformat.write(nb_fixed[1], file)