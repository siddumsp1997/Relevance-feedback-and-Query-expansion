#Import the necessay packages
import sys  
import os
import nltk
import re
import time
from nltk.tokenize import word_tokenize
import lucene
from os import path, listdir
import numpy
import math
from scipy import spatial
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
directory = RAMDirectory()

# Get and configure an IndexWriter
analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
analyzer = LimitTokenCountAnalyzer(analyzer, 250000)
config = IndexWriterConfig(Version.LUCENE_CURRENT, analyzer)

writer = IndexWriter(directory, config)

for input_file in listdir(INPUT_DIR): # iterate over all input files
	#print "Current file:", input_file
	doc = create_document(input_file) # call the create_document function
	writer.addDocument(doc) # add the document to the IndexWriter

writer.close()




#Word vector 
word_vector = {}

#List of docs from lucene search
lucene_output_docs = {}



def lucene_search_loop(searcher, analyzer, query_list):

	#opening the query file

	#reading every query from the input file
	for command in query_list:

		x = word_tokenize(command)

		query_no = int(x[0])

		lucene_output_docs[query_no] = []

		temp_q = command

		temp_q = temp_q[5:]

		temp_q = temp_q.lower()

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
				
		precision_recall_output.write(str(q_no) + " " + str(recall) + " " + str(precision) + " "+str(f_measure)+"\n")
		#print(str(q_no) + " Recall = " + str(recall) + " Precision = " + str(precision) + " F-score = "+str(f_measure)+"\n")


	sum_of_precision = 1.0*sum_of_precision / i
	sum_of_recall = 1.0*sum_of_recall / i
	sum_of_fscore = 1.0*sum_of_fscore / i

	precision_recall_output.write("Average values :\n")
	precision_recall_output.write(str(sum_of_recall) + " " + str(sum_of_precision) + " "+str(sum_of_fscore)+"\n")

	original_output.close()
	predicted_output.close()    
	precision_recall_output.close()

#End of the recall precision function





#Vector addition function
def add_vector(cur_query_vector, word):

	word = word.lower()
	cur_word_vector = word_vector[word]

	for i in range(0,300):
		
		cur_query_vector[i] += cur_word_vector[i]


	return cur_query_vector
#End of function




#Getting cosine values of each word with the query
def get_cosine_values(cur_query_vector):

	words_with_cosine_values = {}

	for cur_word in word_vector:

		cur_vector = word_vector[cur_word]

		result = 1 - spatial.distance.cosine(cur_query_vector, cur_vector)

		words_with_cosine_values[cur_word] = result


	return words_with_cosine_values

# End of function






#main function
if __name__ == '__main__': 


	#Loading the corpus vocabulary from the pickle file (generated in step 1 of task 1)

	#Loading the word dictionary from the pickle file
	file_opener = open("vocabulary.p","r")
	words_database = pickle.load(file_opener)
	file_opener.close()

	#print "Vocabulary loaded....\n"

	# Create a searcher for the above defined Directory
	searcher = IndexSearcher(DirectoryReader.open(directory))

	# Create a new retrieving analyzer
	analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)


	start_time = time.time()

	file_data = open("glove.840B.300d.txt","r")

	line = file_data.readline()

	#Date retrieval from the glove file
	for line in file_data:

		tmp_line = line

		word_list = tmp_line.split(" ")

		word = word_list[0]

		word = word.lower()

		if word in words_database:

			word_vector[word] = []

			l = len(word_list)

			for i in range(1,l):

				x = word_list[i].strip("\n")
				x = x.strip("\r")
				x_float = float(x)

				word_vector[word].append(x_float)

			#End of inner for loop	

	#End of outer for loop

	end_time = time.time()
	file_data.close()

	#print "Word vectors generated. Time taken = "+str(end_time -start_time)+" secs"
	#print " Size of the vocabulary = "+str(len(word_vector)) +"\n\n"
	


	# Generating a 300 sized list of values 0
	empty_list = []

	i = 0
	while True:
		empty_list.append(0.00)
		i += 1
		if i == 300:
			break


	expanded_query_output = open("Expanded query.txt","a")
	query_vector_output  = open("query_vector.txt","a")


	# Query processing 
	query_data = open("query.txt","r")

	print("Query vectors are as follows :\n")

	updated_query_list = []

	for line in query_data:

		line = line.strip("\n")
		line = line.strip("\r")
		line = line.strip()

		word_set = line.split(" ")

		query_vector = empty_list

		l = len(word_set)

		#Getting the query vector from the word vector 
		for i in range(1,l):

			#print word_set[i]
			#query_vector = add_vector(query_vector, "you")
			#print query_vector
			#print "cur word = "+word_set[i]

			word_set[i] = word_set[i].strip()

			if word_set[i] in word_vector:
				query_vector = add_vector(query_vector, word_set[i])

		#End of for loop


		# query_vector_output.write("Query = "+line)
		# query_vector_output.write(query_vector)
		# query_vector_output.write("-------------------------------------------------------\n")

		print("Query = "+line+"\n")
		print(query_vector)
		print("\n-------------------------------------------------------------------------------------------------------------------------------\n\n")

		#Getting the cosine value of the words wrt the given query
		words_with_cosine_values = get_cosine_values(query_vector)	

		#Sorting the dictionary in descending order of cosine similarity values
		sorted_dict = sorted(words_with_cosine_values, key=words_with_cosine_values.get, reverse=True)

		k = 0

		# Picking up the top 5 words and appending it to the query
		new_query = line 

		for r in sorted_dict:

			new_query += " " + str(r)

			k += 1

			if k == 5:
				break

		# print "Original query = "+ line
		# print "Expanded query = "+ new_query	
		# print "-----------------------------------------------------------------------\n"	

		expanded_query_output.write("Original query = "+ line+"\n")
		expanded_query_output.write("Expanded query = "+ new_query+"\n")
		expanded_query_output.write("----------------------------------------------------------------------------------------------\n")

		updated_query_list.append(new_query)


	#End of query for loop



	#Closing all the files
	query_vector_output.close()
	expanded_query_output.close()
	query_data.close()	

	#Lucene search for retrieving top 50 documents
	lucene_search_loop(searcher, analyzer, updated_query_list)	

	#print "Lucene results updated to lucene_output.txt !!"

	recall_precision_calc("lucene_output.txt","Performance_after_query_expansion.txt")

	#print "Precision-recall results updated to Performance_after_query_expansion.txt !!"



#End of main function


		









