#!/usr/bin/python -tt
"""
author: Kai Huang
date:   Feb 7th, 2015

        The main.py tests the decision tree learning and post pruning algorithms. It builds 
        a decision tree on training data, prunes the decision tree using validation data, and
        prints the final pruned decision tree with its prediction accuracy on test data.

"""

import sys
from decisionTree import DecisionTree
from validator import Validator
    
def main():
	
	# Check if the number of command line arguments is correct
	if len(sys.argv) < 6:
	  print("-- python --|-- main.py --|-- L --|-- K --|-- training_set.csv --|-- validation_set.csv --|-- test_set.csv --|")
	  sys.exit(1)

	# The program takes two integer L and K as input to prune the decision tree
	L = int(sys.argv[1])
	K = int(sys.argv[2])

	# Get the file path of the training data, validation data and test data
	dataDir = './data1/'
	training_set = dataDir + sys.argv[3]
	validation_set = dataDir + sys.argv[4]
	test_set = dataDir + sys.argv[5]

	# Build a decision tree on training data
	decisionTree = DecisionTree(training_set)

	#############################################
	# decisionTree.exportTree('tree.txt') 
	# print decisionTree      
	#############################################

	# Create a validator using test data to calculate the prediction accuracy of a given decision tree 
	validator = Validator(test_set)

	# Calculate the prediction accuracy of the original decision tree on test data
	validator.calculateAccuracy(decisionTree.root)

	# Display the prediction accuracy before pruning
	print "\nA decision tree is fully grown to fit the training data." 
	validator.displayAccuracy()

	# Post pruning the decision tree
	print "\nPost prunning", '.' * 30
	print "L =", L, ", K =", K, ", the pruned decision tree is:\n"

	# Prune the original decision tree using L, K and validation data as inputs
	decisionTree.pruneTree(L, K, validation_set)

	##############################################
	# decisionTree.exportTree('pruned_tree.txt')
	##############################################
    
	# print the decision tree to standard output
	print decisionTree  # Override the __str__ method in DecisionTree class

	# Calculate the prediction accuracy of the pruned decision tree on test data
	validator.calculateAccuracy(decisionTree.root)

	# Display the prediction accuracy after pruning
	validator.displayAccuracy()

# Boiler plate function
if __name__ == '__main__':
    main()

