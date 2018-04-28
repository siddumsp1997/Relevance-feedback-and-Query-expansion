# Relevance-feedback-and-Query-expansion

Pseudo Relevance Feedback :
1. Represent each query and document as tf-idf vector where the corpus will be all the
documents in alldocs.rar merged.
2. For each query first retrieve top 50 documents using pylucene (pylucene code's 
available in my GitHub acc). Report precision, recall, f-measure for each query
as well as the average.
3. Now apply Rocchio algorithm to update each query vector by considering top 10
retrieved documents as relevant ones. (Alpha = 1, Beta = 0.65, Gamma = 0)
4. Now from the updated query vector , pick up the top 10 term to obtain the updated
query.
5. For updated query again retrieve top 50 documents using pylucene. 
Finally, we calculate precision, recall, f-measure for each query as well as the average.


Query Expansion :
1. A GloVe vector file (consider this as global knowledge) is provided where each line
contains a word along with a 300 dimension vector. 
Check this out: https://drive.google.com/open?id=1FICPL4UzoeWJQimPUoS9qXoAtjrQlAEW
2. Represent each query as a vector by adding the word vectors of the words present in the
query.
3. Now find top 5 similar (use cosine similarity) words with query vector from GloVe vector
file. Use these 5 words to expand the already existing query.
4. For each expanded query retrieve top 50 documents using pylucene. 
Finally, we calculate precision, recall, f-measure for each query as well as the average.


You can get the datasets here : https://drive.google.com/open?id=1l4gZR7f7GpffEPXqafabkrEGn512F7IJ
Dataset Description:
1. query.txt contains total 82 queries, which has 2 columns query id and query.
2. alldocs.rar contains documents file named with doc id. Each document has set of sentences.
3. output.txt contains 50 relevant documents (doc id) for each query



