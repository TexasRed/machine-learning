#!/usr/bin/python -tt
"""
author: Kai Huang
date:   Feb 7th, 2015
"""
import csv

class CsvParser:

	"""Parse a csv file that contains the data set and retrieve the attribute names, 
	   attribute values, and class values in the data set.

	Attributes:
		attributeNames  : A list of all the attribute names in the data set.
		data            : A matrix of the attribute values in the data set.
		attributes      : A list that records the column index of attributes in the data set.
		examples        : A list that records the row index of the example instances in the data set
		targetAttribute : A vector of the class values in the data set.

	"""
	def __init__(self, filename):

		"""Populate the CsvParser with the attribute names, attribute values, 
		   and class values in the data set.

		Args:
		    filename : The filename of the data set to be parsed.

		"""

        # Create a matrix to store the attribute values
		self.data = []

		# Retrieve attribue names and values from the csv file
		with open(filename,'rb') as csvfile:
			csvreader = csv.reader(csvfile, delimiter = ',')
			count = 0
			for row in csvreader:
				# Retrieve the attribute names from the header of the csv file 
			    if count == 0:
			        self.attributeNames = row[:-1]
			    # Retrieve the attribute values in the following rows
			    else: 
			        self.data.append([int(i) for i in row])
			    count += 1

	    # Create a list to record the column index of the attributes
		self.attributes = range(len(self.attributeNames))

		# Create a list to record the row index of the example instances
		self.examples = range(len(self.data))

		# Create a vector of the class values in the data set
		self.targetAttribute = [row[-1] for row in self.data]
