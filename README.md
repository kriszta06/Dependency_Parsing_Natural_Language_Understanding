# Transition-Based Dependency Parser (Arc-Eager) – AI Model

**Goal:**  
`MODEL AI = INPUT sentence + OUTPUT dependency tree -> dependency_tree_in_CoNLL-U`

---

## 1. Data Loading
- Read the CoNLL-U files using `conllu_reader.py`  
- **Note:** Keep the exact token forms

---

## 2. Filtering
- Remove non-projective sentences  
- **Optional:** Track how many sentences are eliminated for statistics

---

## 3. Conversion to Objects
- Parse CoNLL-U lines into `Token` / `Sentence` objects  
- **Check:** IDs are from 1 to N and a Root exists in the State

---

## 4. Vocabulary Dictionary
- Prepare dictionaries for `FORM`, `UPOS`, `DEPREL`  
- Build mappings:  
  - `word2id` (PAD=0, UNK=1)  
  - `upos2id` (PAD)  
  - `deprel2id`  
  - `action2id`  
- **Note:** UNK for unseen words in dev/test

---

## 5. Example Generation (Oracle)
- Implement Oracle using **Arc-Eager**  
- Receives the gold tree  
- Generates the transition sequence that reconstructs the tree  
- Each `(state, action)` = a training example  
- **Rules:** Oracle must respect preconditions from slides  
- **Example:** `(state, gold_action[, gold_deprel]) => (current_state, RIGHT-ARC(nsubj))` → add to dataset  
- Arc-Eager executes the action decided by the Oracle

---

## 6. Feature Extraction
- Use `state_to_feats()` to convert a `State` into feature IDs  
- Example: top-2 stack + top-2 buffer = words and UPOS  
- For each position, extract `word_id` and `upos_id` (PAD if nonexistent)  
- **Example features:**  
```text
[w_s1, w_s2, w_b1, w_b2, pos_s1, pos_s2, pos_b1, pos_b2]
```

---

## 7. Vectorization

- Features: word embedding + POS embedding

- Concatenate embeddings for model input

---

## 8. Model (Feed-Forward)

- Input layers for indices → embeddings → concatenation → Dense(hidden, ReLU)

- Final Dense layers:

  - Softmax for action prediction

  - Softmax for DEPREL prediction

---

## 9. Training

- Train on train.conllu, dev.conllu, test.conllu (cleaned)

- Loss: cross-entropy on actions

- Batch training

- Monitor dev loss/accuracy

- Save model + vocabularies

---

## 10. Parsing

- For each test sentence:

  - Start from initial state: `stack = [0], buffer = [1…N], arcs = {}`

  - Extract features: state_to_feats(state)

  - Predict: `probs_action, probs_deprel = model.predict(feats)`

  - Validate preconditions → sort by probability → choose first valid action

  - If action is LEFT/RIGHT → label = argmax(probs_deprel)

  - Apply transition

- **Note:** Vertical decoding

---

## 11. Post-Processing (postprocessor.py)

- Repair invalid trees after parsing

- Token without a parent → assign ROOT or nearest candidate

- Multiple ROOTs → keep only one

---

## 12. Output Saving

- Save in CoNLL-U format with HEAD/DEPREL

- Keep other columns unchanged

---

## 13. Evaluation

- Run conll18_ud_eval.py on gold test → compute LAS / UAS

---

## Arc-Eager Parser

- Transition-based parser: stack, buffer, arcs

- Actions: SHIFT, LEFT-ARC, RIGHT-ARC, REDUCE

- arcs contains the final tree

---

## Oracle

- Analyzes current state → decides next Arc-Eager transition

- Receives gold tree

- Simulates Arc-Eager → chooses correct (gold) action

---

## Gold Tree

- Correct tree from CoNLL-U

---

## Token Format (CoNLL-U Line)

- ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC

- **Notes:**

  - LEMMA / XPOS / FEATS / DEPS / MISC → not used

  - HEAD = parent ID

  - DEPREL = dependency relation

---

## Evaluation Metrics

- UAS: Correct HEAD prediction per word

- LAS: Correct HEAD + DEPREL prediction (parent + relation)
