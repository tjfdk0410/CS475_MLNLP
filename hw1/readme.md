# Bag-of-Words Classification with scikit-learn

In this task, you will implement the Bag-of-Words model and text classification in Python.

Implement two methods:
- `preprocess_and_split_to_tokens(sentences) -> tokens_per_sentence`
- `create_bow(sentences, vocab, msg_prefix) -> (vocab, bow_array)`

## Instruction
* See skeleton codes below for more details.
* Do not remove assert lines and do not modify methods that start with an underscore.
* Do not use the bag-of-words function implemented in scikit-learn.
* Before submit your code in KLMS, please change the name of the file to your student id (e.g., 2019xxxx.py).
* Functionality and prediction accuracy for unknown test samples (i.e., we do not give them to you) will be your grade.
* For functionality, we will run unit tests of `preprocess_and_split_to_tokens` and `create_bow`.
* For prediction accuracy, if it is on par with the score of TA, you will get a perfect score.
* TA's validation accuracy is 0.861.
* See https://scikit-learn.org/stable/modules/classes.html for more information.