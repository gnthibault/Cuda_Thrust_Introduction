/*
 * main.cu
 *
 *  Created on: Jun 15, 2015
 *      Author: Thibault Notargiacomo
 */




//STL
#include <cstdlib>

//Local
#include "Optimisation.cu.h"


int main( int argc, char* argv[] )
{
	testCublasWrapper();
	testVariationalSignalDenoising();
	return EXIT_SUCCESS;
}



