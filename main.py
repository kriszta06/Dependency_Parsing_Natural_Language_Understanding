from conllu_reader import ConlluReader
from algorithm import ArcEager
from evaluate import final
from model import ParserMLP
from postprocessor import PostProcessor
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


train_samples = []
dev_samples = []
for sentence in train_trees:
    train_samples.append(arc_eager.oracle(sentence))
for sentence in dev_trees:
    dev_samples.append(arc_eager.oracle(sentence))

model = ParserMLP()
model.train(train_samples, dev_samples)

actionAcc, labelAcc = model.evaluate(dev_samples)

eval_path = "Output/dev_evaluation.txt"
with open(eval_path, "w", encoding="utf-8") as f:
    f.write("=== DEVELOPMENT SET EVALUATION ===\n")
    f.write(f"Action Accuracy: {actionAcc:.4f}\n")
    f.write(f"Label Accuracy:  {labelAcc:.4f}\n")

# ============================================================
# 4. INFERENCE on TEST SET
#    Predict transitions for each test sentence
# ============================================================

predicted_transitions = [
    model.predict_transitions(sentence)
    for sentence in test_trees
]

# ============================================================
# 5. Convert predicted transitions → dependency trees
# ============================================================

predicted_states = [
    model.transitions_to_tree(sentence, transitions)
    for sentence, transitions in zip(test_trees, predicted_transitions)
]

# ============================================================
# DEBUG: Show first 5 predicted arc sets
# ============================================================

print("\n===== DEBUG: Checking predicted states =====\n")

for i, state in enumerate(predicted_states[:5]):
    print(f"Sentence {i}:")
    print(f"  Number of predicted arcs: {len(state.A)}")
    print("  Arcs:", state.A)
    print()

# ============================================================
# 6. Write raw predictions to .conllu
# ============================================================

path = "Output/prediction.conllu"
final(predicted_states, test_trees, path)

# ============================================================
# 7. POSTPROCESS invalid trees
# ============================================================

post = PostProcessor()
clean_trees = post.postprocess(path)

# ============================================================
# 8. Save cleaned output
# ============================================================

output_clean = "Output/prediction_clean.conllu"

with open(output_clean, "w", encoding="utf-8") as f:
    for tree in clean_trees:
        for tok in tree[1:]:  # skip fake root
            f.write(str(tok) + "\n")
        f.write("\n")

print("\nPipeline finished successfully!")

# run "python conll18_ud_eval.py en_partut-ud-test_clean.conllu Output/prediction_clean.conllu -v" and check the final results for a debait!


# TODO: Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.


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

