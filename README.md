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
