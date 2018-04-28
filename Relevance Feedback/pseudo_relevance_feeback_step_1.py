import nltk
import os
import numpy
import re
import sys
import math
import pickle

# Class to store document (node) information
class node_data:

	# A node contains sentence
	def __init__(self,str_sentence):

		self.sentence = str_sentence
		self.tf = {}
		self.idf = {}		




#vocabulary
words_database = {}
doc_id  = {}



# # Processing line by line
# def word_by_word_processing(line):
	
	
# 	line = re.sub("[^a-zA-Z]+", " ", line)

# 	set_of_stop_words = nltk.corpus.stopwords.words('english')

# 	#stem_ob = nltk.stem.porter.PorterStemmer()

# 	#tokenset = nltk.tokenize.word_tokenize(line)

# 	#final_set_of_tokens = []

# 	final_line = ""

# 	for token in tokenset:

# 		token = token.lower()

# 		if token not in set_of_stop_words:

# 			#final_line += stem_ob.stem(token)
# 			final_line += token
# 			final_line += " "       
		
# 	return final_line




# This text processing module is wrt to every doc in the folder "alldocs"
def Doc_processing_module():


	file_node_list = []

	#file_list = os.listdir("/home/sid/Downloads/Assignement2_IR/Topic"+str(i+1))
	file_list = os.listdir("./alldocs")

	temp_tkn = nltk.data.load('tokenizers/punkt/english.pickle')

	i = 0
	#iterate thru every file
	for file in file_list:

		#print file

		doc_id[file] = i
		#file_ob = open("/home/sid/Downloads/Assignement2_IR/Topic"+str(i+1)+"/"+file,"r")
		file_ob = open("./alldocs"+"/"+file,"r")

		# concatenating all the text in the folder to one entity
		file_text = file_ob.read()

		#final_text = word_by_word_processing(file_text)
		#print file_text + "\n\n\n"

		node = node_data(file_text)
		file_node_list.append(node)

		i += 1


	with open("doc_id_data.p","wb") as doc1_data:
		pickle.dump(doc_id, doc1_data)	

	return file_node_list		

#End of function


# This text processing module is wrt to every query in "query.txt"
def Query_processing_module():

	query_node_list = []

	#file_list = os.listdir("/home/sid/Downloads/Assignement2_IR/Topic"+str(i+1))
	queries = open("query.txt",'r')

	#temp_tkn = nltk.data.load('tokenizers/punkt/english.pickle')

	#iterate thru every file
	for query in queries:

		node = node_data(query)
		query_node_list.append(node)


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
def generate_tf_idf_vectors(node_list):


	# Dictionary for storing the entire vocabulary
	# Vocabulary stores the no of nodes in which a 
	# particular word appears


	#Calculation of tf
	for node in node_list:

		word_list = getwordlist(node)

		word_set = set(word_list)
		
		for word in word_set:

			node.tf[word]  = 0

			if word not in words_database:

				words_database[word] = 1

			else:

				words_database[word] += 1

		#finding out the tf-vector of the node
		for word in word_list:

			node.tf[word] += 1	

	#Calculation of idf

	i = 0

	N = len(words_database)

	nodes_to_be_removed = []

	for node in node_list:

		word_list = getwordlist(node)

		word_set = set(word_list)

		if len(word_set) == 0:

			nodes_to_be_removed.append(i)

		for word in word_set:

			ni = words_database[word]
			#print "word = "+ word + "  N = "+ str(N)+ " ni = "+str(ni)
			node.idf[word] = math.log(N*1.0/ni)


		i = i + 1		

	#end of for loop
	
	#print("size of nodes to be removed =  "+str(len(nodes_to_be_removed)))	


	#Removing invalid nodes (nodes containing invalid elements)

	# final_node_list = []

	# l = len(node_list)

	# for i in range(0,l):

	# 	if i not in nodes_to_be_removed:

	# 		final_node_list.append(node_list[i])


	with open("doc_data.p","wb") as doc_data:
		pickle.dump(node_list, doc_data)					


	return node_list	

# End of function




# Module to generate tf-idf vectors corresponding to the sentences
def generate_tf_idf_vectors_for_query(node_list):


	#Calculation of tf
	for node in node_list:

		word_list = getwordlist(node)

		#print str(word_list[0])+" is the indeccs"

		#wordlist.pop(0)

		word_set = set(word_list)
		
		for word in word_set:

			node.tf[word]  = 0

		#finding out the tf-vector of the node
		for word in word_list:

			node.tf[word] += 1	

	#Calculation of idf

	i = 0

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
				node.idf[word] = 10000	

		i = i + 1		

	#print("Size of vocabulary : "+str(len(words_database))+"\n\n")			

	return node_list	

# End of function



#main function
if __name__ == '__main__':  
	
	#Documents processing and tf vector and idf vector generation
	doc_list = Doc_processing_module()

	doc_node_list = generate_tf_idf_vectors(doc_list)

	#Query processing and tf vector and idf vector generation
	query_list = Query_processing_module()

	query_node_list = generate_tf_idf_vectors_for_query(query_list)

	#Storing word vocabulary in a file
	with open("vocabulary.p","wb") as voc_data:
		pickle.dump(words_database, voc_data)


