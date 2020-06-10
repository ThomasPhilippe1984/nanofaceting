/************************************************
* Philippe Thomas								*
* Monguillon Julien								*
* L.PMC - CNRS / Ecole Polytechnique			*
************************************************/

#ifndef _NANOFACETING_H_
#define _NANOFACETING_H_

#define STRING_LENGTH (size_t)128	// Parameter max length

// Standard C library
#include <cstdlib>					// exit(), EXIT_SUCCESS & EXIT_FAILURE
#include <ctype.h>					// ispunct()
#include <cmath>					// pow(), sqrt()
#include <time.h>					// clock_t, clock(), CLOCKS_PER_SEC, time_t, time(), localtime()
#include <string.h>					// strcmp(), memset()

// Standard C++ library
#include <fstream>					// ofstream
#include <iostream>					// cout
#include <sstream>					// stringstream
#include <algorithm>				// count(), transform()

// Class for manipulate 2D arrays
#include "CArray2D.hpp"

// Definitions for the model, given in the parameters file, in double because parameters parsing has to manage double values
#define INIT_METHOD_BASE			(int)0		// Counter start for initialization method choices
#define INIT_METHOD_PLAN_SMOOTH		(int)1		// Initialization with planar front, noisy and smooth with tanh
#define INIT_METHOD_PLAN_ABRUPT		(int)2		// Initialization with planar front, noisy but abrupt
#define INIT_METHOD_SPHERE			(int)3		// Initialization with a spherical interface
#define MODEL_TYPE_BASE				(int)10		// Counter start for model type choices
#define MODEL_TYPE_SIMPLE			(int)11		// Simple model
#define MODEL_TYPE_COMPLETE			(int)12		// Complete model
#define RESSOURCE_USED_BASE			(int)100	// Counter start for ressource choices
#define RESSOURCE_USED_CPU			(int)101	// Use CPU only, one core at a time
#define RESSOURCE_USED_CPU_MT		(int)102	// Use CPU only, multiple cores at the same time
#define RESSOURCE_USED_GPU			(int)103	// Use GPU mostly
#define PRECISION_TYPE_BASE			(int)1000	// Counter start for precision type choices
#define PRECISION_TYPE_SP			(int)1001	// Use single precision (SP) - float
#define PRECISION_TYPE_DP			(int)1002	// Use double precision (DP) - double

// Ghost points management
#define GHOST_POINTS_CPU_SIMPLE		(int)2			// Ghost points for the simple model on CPU
#define GHOST_POINTS_CPU_COMPLETE	(int)4			// Ghost points for the complete model on CPU
#define GHOST_POINTS_GPU_COMPLETE	(int)0			// Ghost points for the complete model on GPU

// Simulation parameters
typedef struct
{
	// System size
	uint64_t MX;					// Box's size in X
	uint64_t MY;					// Box's size in Y
	// Constants
	double delta;					// Interface thickness
	double a;						// Gamma function's parameter
	// Simulation precision and length
	double dx;						// delta-X
	double dt;						// delta-T
	uint64_t nbrTStep;				// Used to calculate simulation's duration
	uint64_t TStepsByDump;			// Number of time steps between two data dumps
	// Constants specific to the complete model
	double noise;					// Noise amplitude (thermal fluctuation)
	double deltamu;					// Driving force for crystallization
	double omega;					// Gamma function's parameter
	double alpha;					// Corner size
	double b;						// Constant in the term for driving force for growth (if growth activated)
	// Simulation properties
	int initMethod;					// Initialization method
	int modelType;					// Type of the model : see upper definitions section to know the different possible values
	int resourceUsed;				// Resource used : CPU (with one thread or multiple with OpenMP) or GPU
	int precisionType;				// Precision : single (float) or double
} simulationParameters_t;

// Program parameters
typedef struct
{
	simulationParameters_t	p;				// Simulation parameters
	uint64_t  				GHOST_POINTS;	// because of second derivative
	uint64_t				MXG;			// box's size in X with ghost points
	uint64_t				MXGH;			// box's size in X with ghost points (half quantity)
	uint64_t				MYG;			// box's size in Y with ghost points
	uint64_t				MYGH;			// box's size in Y with ghost points (half quantity)
	CArray2D<double>		u1;				// Model's phasefield
	CArray2D<double>		u2;				// Model's phasefield (t+1)
	CArray2D<double>		*us;			// Array's pointer : source
	CArray2D<double>		*ud;			// Array's pointer : destination
	// Optimization variables for the simple model
	double					factor1;
	double					factor2;
	double 					invdxp2m12;
	// Specific to the complete model : phase arrays used to avoid redundant computations
	CArray2D<double>		theta;			// Orientation
	CArray2D<double>		w;				// W function of phasefield model
	// Optimization variables for the complete model
	double 					dxddelta;
	double 					invdxm2;
	double					deltap2;
	double					dtddeltap2;
	double					factor3;
	double					factor4;
	double					noisem2;
} simulationData_t;

/* Function declarations */
bool isParameterLineCorrect   (std::string);
void printParameters          (simulationParameters_t);
int  parseParameters          (simulationParameters_t*, int, char**);
int  initSimulationParameters (simulationParameters_t*);
int  initSimulationData       (simulationData_t*);
int  initData                 (simulationData_t*);
int  resetData                (simulationData_t*);
int  writeData			      (simulationData_t*, uint64_t);
int  swapArrayPointers        (simulationData_t*);
int  swapArrayPointersFloat   (float**, float**);
int  swapArrayPointersDouble  (double**, double**);

#endif	// ndef _NANOFACETING_H_
