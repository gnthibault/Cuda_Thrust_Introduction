#Sys for main args
import sys, getopt

#import/export in csv format
import csv

#For plotting results
import matplotlib.pyplot as plt 

#numpy for array, matrix, ...
import numpy as np

def main(argv):
	####################################
	###        Parse arguments       ###
	####################################
	args = str( sys.argv )
    
	print" Usage is: ./printCurves.py ./data.csv "
    
	#Input and output fileName
	dataCSV = str( sys.argv[1] ) #the csv file
 
	####################################
	##            Import data        ##
	####################################
	#Import from dataCSV
	data = np.genfromtxt( dataCSV, delimiter=',')

	fig = plt.figure(1)
	plt.plot( data[0,:], label="Reference" )
	plt.plot( data[1,:], label="Noisy (5%)" )
	plt.plot( data[2,:], label="Denoised" )
	plt.legend(loc=1,prop={'size':8})
	plt.xlabel('Time in s')
	plt.ylabel('Value (no unit)')
	plt.title('Variational signal denoising in 1D')
	plt.grid(True)
	fig.patch.set_facecolor('grey')
	#plt.savefig('VariationalSignalDenoising.png', facecolor=fig.get_facecolor(), edgecolor='none')
	plt.show()


############################################################################
main(sys.argv[1:])
