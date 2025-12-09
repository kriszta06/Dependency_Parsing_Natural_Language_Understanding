
from conllu_reader import ConlluReader
from algorithm import ArcEager

def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees


"""
Read and convert CoNLLU files into tree structures
"""
# Initialize the ConlluReader
reader = ConlluReader()
train_trees = read_file(reader,path="en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="en_partut-ud-test_clean.conllu", inference=True)

"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""
train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))

#Create and instance of the ArcEager
arc_eager = ArcEager()

print ("\n ------ TODO: Implement the rest of the assignment ------")

# TODO: Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.

train_samples = []
for sentence in train_trees:
    train_samples.append(arc_eager.oracle(sentence))

# TODO: Implement the 'state_to_feats' function in the Sample class.
# This function should convert the current parser state into a list of features for use by the neural model classifier.

# TODO: Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training dataset.
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.

# TODO: Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.


"""
Below we will have described the architecture for the steps needed to be impemented
in order to extract neede dataset(gold transitions) from parsing of each sentence
using oracle function(arc eager alghoritm).
For each sentence:
1.we start with the initial state
2.we iterate thorugh steps while we got to final state
3.at each iteration of parsing we get the gold action with oracle function from arceager class
4.transofrm it to features for our model
5.append it in the dataset variable
(same for train and dev)

"""
"""
####################
# FOR TRAIN DATASET#
####################
train_dataset = []

        for sent in train_trees:
            gold_samples = arc_eager.oracle(sent)

            for sample in gold_samples:
                feats = sample.state_to_feats()
                gold_action = sample.transition.action
                train_dataset.append((feats, gold_action))

####################
# FOR DEV DATASET#
####################
dev_dataset = []

        for sent in dev_trees:
            gold_samples = arc_eager.oracle(sent)
            for sample in gold_samples:
                feats = sample.state_to_feats()
                gold_action = sample.transition.action
                dev_dataset.append((feats, gold_action))
"""

"""

"""