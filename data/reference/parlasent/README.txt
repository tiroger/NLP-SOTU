The multilingual sentiment dataset of parliamentary debates ParlaSent 1.0
http://hdl.handle.net/11356/1868

The dataset consists of five training datasets and two test sets. The test sets have a _test.jsonl suffix.

The attributes in training data are the following:
- sentence - the sentence labeled for sentiment
- country - the country of the parliament the sentence comes form
- annotator1 - first annotator's annotation
- annotator2 - second annotator's annotation
- reconciliation - the final label agreed upon after reconciliation
- label - three level (positive, negative, neutral) label based on the reconciliation label
- document_id - internal identifier of the document the sentence comes form
- sentence_id - internal identifier of the sentence inside the document
- term - the term of the parliament the sentence comes from
- date - the date the sentence was uttered as part of a speech in the parliament
- name - name of the MP giving the speech
- party - the party of the MP
- gender - binary gender of the MP
- birth year - year of birth of the MP
- split - whether the sentence is to be used as a training, development or testing instance in case evaluation is done of the training portion of the dataset
- ruling - whether the MP was in a coalition or an opposition at the time of giving the speech

The attributes in the test data (_test.jsonl files) are the following:
- sentence - the sentence labeled for sentiment
- country - the country of the parliament the sentence comes form
- annotator1 - first (only) annotator's annotation, used as a final annotation
- label - three level (positive, negative, neutral) label based on the annotator1 label
- document_id - internal identifier of the document the sentence comes form
- sentence_id - internal identifier of the sentence inside the document
- term - the term of the parliament the sentence comes from
- date - the date the sentence was uttered as part of a speech in the parliament
- name - name of the MP giving the speech
- party - the party of the MP
- gender - binary gender of the MP
- birth year - year of birth of the MP
- ruling - whether the MP was in a coalition or an opposition at the time of giving the speech
