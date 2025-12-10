def final(predicted_states, test_trees, path):


    """
    Writes the predicted dependency trees to a CoNLL-U file.

    predicted_states: output from model.run(test_trees)
    test_trees: the original token structures from the test set
    path: output file path (string)
    """

    with open(path, "w", encoding="utf-8") as f:
        for sent, state in zip(test_trees, predicted_states):

            # Construct head[] and deprel[] from predicted arcs
            n = len(sent)
            head = [0] * (n + 1)
            deprel = ["_"] * (n + 1)
            unique_arcs = {}

            for (h, lbl, dep) in state.A:
                # Skip impossible deps
                if dep < 1 or dep > n:
                    continue
                if h < 0 or h > n:
                    continue

            for (h, lbl, dep) in unique_arcs.items():
    # Skip invalid arcs
                head[dep] = h
                deprel[dep] = lbl


            # Write tokens in CoNLL-U
            for tok in sent:
                f.write(
                    f"{tok.id}\t"
                    f"{tok.form}\t"
                    f"{tok.lemma}\t"
                    f"{tok.upos}\t"
                    f"{tok.cpos}\t"      
                    f"{tok.feats}\t"
                    f"{head[tok.id]}\t"
                    f"{deprel[tok.id]}\t"
                    f"{tok.deps}\t"
                    f"{tok.misc}\n"
                )

            f.write("\n")
