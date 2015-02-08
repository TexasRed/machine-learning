#!/usr/bin/python -tt
"""
author: Kai Huang
date:   Feb 7th, 2015

"""

from csvParser import CsvParser

class Validator:

	"""Create a validator to calculate the prediction accuracy of a given decision tree 
	   on validation data or test data.

	Attributes:
	    targetAttribute : The class value vector in the data set.
	    data            : The attribute value matrix in the data set.
	    
	"""

	def __init__(self, filename):

		"""Initialize the validator by populating the validator with all the attribute values 
		   and class values from the data set.

		Args:
		    filename : the filename of the data set

		"""
        # Parse the csv file that contains the data set
		csvParser = CsvParser(filename)
        
        # Get the targetAttribute value vector from the data set
		self.targetAttribute = csvParser.targetAttribute

        # Get the attribute value matrix from the data set
		self.data = csvParser.data

	def calculateAccuracy(self, root):

		"""Calculate the prediction accuracy of a given decision tree on the data set.

		Args:
		    root : the root TreeNode of the decision tree.

		Returns:
		    The prediction accuracy on data set.

		"""

		# If the decision tree or the data set is empty, return accuracy as 0
		if root == None or len(self.data) == 0:
			return 0

		# Count the total number of correct predictions made by the decision tree
		count = 0
		for i in range(len(self.data)):
			if self.getPredictedValue(root, self.data[i]) == self.targetAttribute[i]:
				count += 1

		# Calculate and return the prediction accuracy
		self.accuracy = 1.0 * count / len(self.data)
		return self.accuracy

	def getPredictedValue(self, root, row):

		"""Given an instance in the data set, return the class value predicted by the decision tree model.

		Args:
		    root : The root TreeNode of the decision tree.
		    row  : The row index of the instance in the data set.

		Returns:
		    The class value predicted by the decision tree model.

		"""

		# Return the predicted class value if reaches at a leaf node
		if root.val == -1:
			return root.label

		# If an attribute value is 0, search in the left subtree
		if row[root.val] == 0:
			return self.getPredictedValue(root.left, row)

		# If an attribute value is 1, search in the right subtree
		else:
			return self.getPredictedValue(root.right, row)

	def displayAccuracy(self):

		"""Display the prediction accuracy of a given decision tree on the data set.

		"""
		
		print "The prediction accuracy on test data = {0:.2f}%".format((self.accuracy) * 100)