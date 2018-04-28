#Import the necessay packages
import sys  
import os
import nltk
import re
from nltk.tokenize import word_tokenize
import lucene
from os import path, listdir
import numpy
import math
import pickle

#from java.io import File
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.util import Version
from org.apache.lucene.store import RAMDirectory, SimpleFSDirectory
import time
# Indexer imports:
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
# from org.apache.lucene.store import SimpleFSDirectory
# Retriever imports:
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser

# ---------------------------- global constants ----------------------------- #

BASE_DIR = path.dirname(path.abspath(sys.argv[0]))
INPUT_DIR = BASE_DIR + "/alldocs/"
#print INPUT_DIR + "\n\n"



# Class to store document (node) information
class node_data:

	# A node contains sentence
	def __init__(self,str_sentence):

		self.sentence = str_sentence
		self.tf = {}
		self.idf = {}		



#Loading the dictionary from the pickle file
x = open("vocabulary.p","r")
words_database = pickle.load(x)
x.close()

#Loading the dictionary from the pickle file
y = open("doc_data.p","r")
doc_node_list = pickle.load(y)
y.close()

#Loading the dictionary from the doc id hashmap
z = open("doc_id_data.p","r")
doc_id = pickle.load(z)
z.close()


# doc_id = {}

# words_database = {}

#List of docs from lucene search
lucene_output_docs = {}

# Queries tf-idf values
query_tf_idf = {}



def create_document(file_name):
	path = INPUT_DIR+file_name # assemble the file descriptor
	file = open(path) # open in read mode
	doc = Document() # create a new document
	# add the title field
	doc.add(StringField("title", input_file, Field.Store.YES))
	# add the whole book
	doc.add(TextField("text", file.read(), Field.Store.YES))
	file.close() # close the file pointer
	return doc



# Initialize lucene and the JVM
lucene.initVM()

# Create a new directory. As a SimpleFSDirectory is rather slow ...
directory = RAMDirectory() # ... we'll use a RAMDirectory!
#directory = SimpleFSDirectory(File("lucene_index/"))
#directory = SimpleFSDirectory("IR_Assignment1/fd")

# Get and configure an IndexWriter
analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
analyzer = LimitTokenCountAnalyzer(analyzer, 250000)
config = IndexWriterConfig(Version.LUCENE_CURRENT, analyzer)

writer = IndexWriter(directory, config)

#print "Number of indexed documents: %d\n" % writer.numDocs()

for input_file in listdir(INPUT_DIR): # iterate over all input files
	#print "Current file:", input_file
	doc = create_document(input_file) # call the create_document function
	writer.addDocument(doc) # add the document to the IndexWriter

#print "\nNumber of indexed documents =  %d" % writer.numDocs()
writer.close()



# This text processing module is wrt to every query in "query.txt"
def Query_processing_module():

	query_node_list = []

	#file_list = os.listdir("/home/sid/Downloads/Assignement2_IR/Topic"+str(i+1))
	queries = open("query.txt",'r')

	#temp_tkn = nltk.data.load('tokenizers/punkt/english.pickle')

	#iterate thru every file
	i = 0

	for query in queries:

		node = node_data(query)
		query_node_list.append(node)
		#query_id
		i += 1

	return query_node_list		

#End of function





# Get word list from given text
def getwordlist(node):

	sent = node.sentence

	#sent = sent[5:]
	sent = sent.lower()
	sent =re.sub("[^a-zA-Z]+"," ", sent)

	#print sent + "\n"

	sent = sent.strip()
	word_list = sent.split(" ")

	stop_words = nltk.corpus.stopwords.words('english')

	#word_list1 = filter(lambda x: x not in stop_words, word_list)
	#word_list1 = [x for x in word_list if x not in stop_words]

	word_list2 = filter(lambda x: x !='', word_list)

	return word_list2

#end of function	





# Module to generate tf-idf vectors corresponding to the sentences
def generate_tf_idf_vectors_for_query(node_list):


	#Calculation of tf
	for node in node_list:

		word_list = getwordlist(node)

		#print word_list

		#print word_list[0] + "in generating tf-idf-vector-forquery"
		#word_list.pop(0)

		word_set = set(word_list)
		
		for word in word_set:

			node.tf[word]  = 0

		#finding out the tf-vector of the node
		for word in word_list:

			node.tf[word] += 1	

	#Calculation of idf

	N = len(words_database)

	nodes_to_be_removed = []

	for node in node_list:

		word_list = getwordlist(node)

		word_set = set(word_list)

		for word in word_set:

			if word in words_database :

				ni = words_database[word]
				#print str(ni) + "\n"
				node.idf[word] = math.log(N*1.0/ni)

			else: 
				node.idf[word] = 100000				


	return node_list	

# End of function




# Function to return lucene search results
def search_loop(searcher, analyzer):

	#opening the query file

	queries = open("query.txt",'r')

	#reading every query from the input file
	for command in queries:

		x = word_tokenize(command)

		query_no = int(x[0])

		lucene_output_docs[query_no] = []

		temp_q = command

		temp_q = temp_q[5:]

		#print "search loop:  "+ temp_q + "\n"

		query = QueryParser(Version.LUCENE_CURRENT, "text", analyzer).parse(temp_q)

		# retrieving top 50 results for each query
		scoreDocs = searcher.search(query, 50).scoreDocs

		# writing output to the file
		output_file2 = open("lucene_output.txt","a")

		for scoreDoc in scoreDocs:

			doc = searcher.doc(scoreDoc.doc)
			#print doc.get("title")#, 'name:', doc.get("name")
			temp_str = str(doc.get("title"))

			lucene_output_docs[query_no].append(temp_str)

			output_file2.write(str(query_no)+" "+temp_str+"\n")

		#Results retrieved	

		output_file2.close()	

	#End of outer for loop


	#Closing the queries file
	queries.close()

# End of function




def modified_search_loop(searcher, analyzer, query_list):

	#opening the query file

	#reading every query from the input file
	for command in query_list:

		x = word_tokenize(command)

		query_no = int(x[0])

		lucene_output_docs[query_no] = []

		temp_q = command

		temp_q = temp_q[5:]

		#print "search loop:  "+ temp_q + "\n"

		query = QueryParser(Version.LUCENE_CURRENT, "text", analyzer).parse(temp_q)

		# retrieving top 50 results for each query
		scoreDocs = searcher.search(query, 50).scoreDocs

		# writing output to the file
		output_file2 = open("lucene_output_for_updated_queries.txt","a")

		for scoreDoc in scoreDocs:

			doc = searcher.doc(scoreDoc.doc)
			#print doc.get("title")#, 'name:', doc.get("name")
			temp_str = str(doc.get("title"))

			lucene_output_docs[query_no].append(temp_str)

			output_file2.write(str(query_no)+"  "+temp_str+"\n")

		#Results retrieved	

		output_file2.close()	

	#End of outer for loop

# End of function




#Recall and precision calculation
def recall_precision_calc(predicted_output_data, filename):


	#filename = "precision_recall_output.txt"
	predicted_output = open(predicted_output_data, "r")

	query = open("query.txt","r")

	original_output = open("output.txt","r")

	predicted_output_table = {}

	predicted_files = []
	query_id1 = []

	original_files = []
	query_id2 = []

	original_output_table = {}

	sum_of_precision = 0.0
	sum_of_recall = 0.0
	sum_of_fscore = 0.0
	
	
	precision_recall_output = open(filename,"a")

	precision_recall_output.write("Scores will be of the following order :\n")
	precision_recall_output.write("  Precision  Recall  F-measure\n")

	query_ID = []

	for line in query:

		x = word_tokenize(line)
		query_ID.append(x[0])


	for line in original_output:

		x = word_tokenize(line)

		if x[0] in original_output_table:

			original_output_table[x[0]].append(x[1])

		else:

			temp_list = [x[1]]

			original_output_table[x[0]] = temp_list	    


	for line in predicted_output:

		x = word_tokenize(line)

		if x[0] in predicted_output_table:

			predicted_output_table[x[0]].append(x[1])

		else:
			temp_list = [x[1]]

			predicted_output_table[x[0]] = temp_list


	i = 0
	    
	for q_no in query_ID:
		
		original = original_output_table[q_no]

		if q_no in predicted_output_table:
			predicted = predicted_output_table[q_no]
		else:
			predicted = []


		recall = len(list(set(original).intersection(set(predicted))))/float(len(original))

		#if len(predicted) != 0:
		precision = len(list(set(original).intersection(set(predicted))))/float(len(predicted))
		f_measure = (2*precision*recall) / (precision + recall)

		# else:
		# 	precision = 0.0
		# 	f_measure = 2*recall


		sum_of_recall += recall
		sum_of_fscore += f_measure
		sum_of_precision += precision	
				
		i += 1
				
		precision_recall_output.write(str(q_no) + "   " + str(recall) + "   " + str(precision) + "   "+str(f_measure)+"\n")
		#print(str(q_no) + " Recall = " + str(recall) + " Precision = " + str(precision) + " F-score = "+str(f_measure)+"\n")


	sum_of_precision = 1.0*sum_of_precision / i
	sum_of_recall = 1.0*sum_of_recall / i
	sum_of_fscore = 1.0*sum_of_fscore / i

	precision_recall_output.write("\nAverage values :\n")
	precision_recall_output.write(str(sum_of_recall) + "   " + str(sum_of_precision) + "   "+str(sum_of_fscore)+"\n")

	original_output.close()
	predicted_output.close()    
	precision_recall_output.close()

#End of the recall precision function






#main function
if __name__ == '__main__':  

	# Create a searcher for the above defined Directory
	searcher = IndexSearcher(DirectoryReader.open(directory))

	# Create a new retrieving analyzer
	analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)

	search_loop(searcher, analyzer)

	#filename = "precision_recall_output.txt"
	#predicted_output = open("lucene_output.txt", "r")

	recall_precision_calc("lucene_output.txt","Performance_before_relevance_feedback.txt")

	#text processing module for retrieving the text from the documents of the folder

	print " Lucene output generated..."

	# doc_list = Doc_processing_module()
	# doc_node_list = generate_tf_idf_vectors(doc_list)
	# print "Doc processing and tf-idf over"


	query_list = Query_processing_module()

	query_node_list = generate_tf_idf_vectors_for_query(query_list)

	print "Query processing and tf-idf over\n"


	updated_query_list = []

	i = 0

	for query_node in query_node_list:

		query_wordlist = getwordlist(query_node)

		#print query_wordlist[0] + " is here"
		#query_wordlist.pop(0)

		query_wordset = set(query_wordlist)

		query_tf_idf[i] = {}

		#for word in words:
		#	query_tf_idf[i][word] = 0.0

		query_no = int(query_node.sentence[0:3])

		#print str(query_no) + " is the current query id\n"


		# Calculating tf-idf vector for the query
		for word in query_wordset:

			if query_node.idf[word] != 100000:

				query_tf_idf[i][word] = math.log(1 + query_node.tf[word]) * query_node.idf[word]


		b_by_delta_dr = 0.065

		# Retrieving only the top 10 documents from lucene output
		j = 0


		# Implementing Rochio algorithm for each query (query vector updation)
		for doc in lucene_output_docs[query_no]:

			str_doc = str(doc)
			doc_index = doc_id[str_doc]

			cur_doc_node = doc_node_list[doc_index]

			doc_word_list = getwordlist(cur_doc_node)

			#print doc_word_list

			doc_word_set = set(doc_word_list)	

			#Rochio algorithm for query vector updation
			for word in doc_word_set:

				if word in query_tf_idf[i]:

					query_tf_idf[i][word] += b_by_delta_dr * math.log(cur_doc_node.tf[word]+1) * cur_doc_node.idf[word]

				else:
				
					query_tf_idf[i][word] = b_by_delta_dr * math.log(cur_doc_node.tf[word]+1) * cur_doc_node.idf[word]	


			j += 1

			#Only top 10 docs from the retrieved 50 docs
			if j == 10:
				break

		    #End of inner for loop



		#Sorting the dictionary entries wrt its Values
		new_query = str(query_no)+"  "
		sorted_dict = sorted(query_tf_idf[i], key=query_tf_idf[i].get, reverse=True)

		k = 0

		for r in sorted_dict:

			new_query += str(r) +" "

			k += 1

			if k == 10:
				break

		print "original query = "+ query_node.sentence
		print "updated query = "+ new_query

		updated_query_list.append(new_query)


		print "---------------------------------------------------------\n\n"

		i += 1	

	#End of outer for loop
	


	#finding out the precision, recall and f-measure for lucene search over the modified queries

	modified_search_loop(searcher, analyzer, updated_query_list)	

	recall_precision_calc("lucene_output_for_updated_queries.txt","Performance_after_relevance_feedback.txt")


#End of main function










