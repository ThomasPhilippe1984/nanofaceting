/************************************************
* Philippe Thomas								*
* Monguillon Julien								*
* L.PMC - CNRS / Ecole Polytechnique			*
************************************************/

// Source file specific library
#include "Nanofaceting.h"

/*
 * Check a line read from a parameter file (in simple text format).
 * Return true if the line is correct (1 equal, 1 possible dot for a non-integer value,
 *   no comma and no punctuation mark).
 * Return false otherwise.
 */
bool isParameterLineCorrect (std::string str) {
	// Test for the pattern "parameter=value" in the line
	if (std::count(str.begin(), str.end(), '=') != 1 ||
	    std::count(str.begin(), str.end(), '.') >  1 ||
		std::count(str.begin(), str.end(), ',') >  0) {
		return false;
	}
	// Test for useless punctuation
	for(size_t i = 0; i < str.length(); ++i) {
		if (ispunct(str[i]) != 0 && str[i] != '=' && str[i] != '.') {
			return false;
		}
	}
	return true;
}

/*
 * Print name and value of parameters to the standard output, one per line
 */
void printParameters (simulationParameters_t p) {
	// String preparations
	std::string initMethodStr = "unknown";
	if (p.initMethod == INIT_METHOD_PLAN_SMOOTH) {
		initMethodStr = "planar front, noisy and smooth with tanh";
	} else if (p.initMethod == INIT_METHOD_PLAN_ABRUPT) {
		initMethodStr = "planar front, noisy but abrupt";
	} else if (p.initMethod == INIT_METHOD_SPHERE) {
		initMethodStr = "spherical interface";
	}
	std::string modelTypeStr = "unknown";
	if (p.modelType == MODEL_TYPE_SIMPLE) {
		modelTypeStr = "simple";
	} else if (p.modelType == MODEL_TYPE_COMPLETE) {
		modelTypeStr = "complete";
	}
	std::string resourceUsedStr = "unknown";
	if (p.resourceUsed == RESSOURCE_USED_CPU) {
		resourceUsedStr = "CPU only, one core at a time";
	} else if (p.resourceUsed == RESSOURCE_USED_CPU_MT) {
		resourceUsedStr = "CPU only, multiple cores at the same time";
	} else if (p.resourceUsed == RESSOURCE_USED_GPU) {
			resourceUsedStr = "GPU mostly";
	}
	std::string precisionTypeStr = "unknown";
	if (p.precisionType == PRECISION_TYPE_SP) {
		precisionTypeStr = "simple (float)";
	} else if (p.precisionType == PRECISION_TYPE_DP) {
		precisionTypeStr = "double";
	}
	// One line per parameter
	std::cout << "-= Simulation parameters =-"						 << std::endl;
	std::cout << "Box's size in X            : " << p.MX			 << std::endl;
	std::cout << "Box's size in Y            : " << p.MY			 << std::endl;
	std::cout << "Interface thickness        : " << p.delta			 << std::endl;
	std::cout << "Constant a                 : " << p.a				 << std::endl;
	std::cout << "Delta-X                    : " << p.dx			 << std::endl;
	std::cout << "Delta-T                    : " << p.dt			 << " sec" << std::endl;
	std::cout << "Number of time steps       : " << p.nbrTStep	     << " (simulation lasts " << p.dt *  p.nbrTStep << " sec)" << std::endl;
	std::cout << "Time steps between 2 dumps : " << p.TStepsByDump   << " (total of " << (int)floor((p.nbrTStep / p.TStepsByDump) + 1.) << " dumps with initial dump at t0)" << std::endl;
	if (p.modelType == MODEL_TYPE_COMPLETE) {
	std::cout << "Noise factor               : " << p.noise 		 << std::endl;
	std::cout << "Delta Mu                   : " << p.deltamu 		 << std::endl;
	std::cout << "Omega                      : " << p.omega			 << std::endl;
	std::cout << "Alpha                      : " << p.alpha			 << std::endl;
	std::cout << "Constant b                 : " << p.b				 << std::endl;
	}
	std::cout << "Initialization method      : " << initMethodStr	 << std::endl;
	std::cout << "Type of model used         : " << modelTypeStr     << std::endl;
	std::cout << "Type of precision used     : " << precisionTypeStr << std::endl;
	std::cout << "Type of resource used      : " << resourceUsedStr  << std::endl;
	std::cout << std::endl;
}

/*
 * Parse program arguments in order to extract simulation parameters.
 * Verify argument's format and completeness of parameters.
 */
int parseParameters (simulationParameters_t *p, int argNbr, char **argStr)
{
    // Argument(s) number check
	if (argNbr != 2) {
        std::cout << "Bad number of arguments, you must specify only a file containing parameters for the simulation" << std::endl;
        exit(EXIT_FAILURE);
	}
    // Opening of file containing parameters
    std::ifstream parameters_file;
    parameters_file.open(argStr[1]);
    if (parameters_file.is_open() != true) {
		std::cout << "Error opening file containing parameters : " << argStr[1] << std::endl;
		exit(EXIT_FAILURE);
	}
	// Reading parameters
	std::string line;
	char parameter_str[STRING_LENGTH];
	double value = 0.0;
	int lineNbr = 1;
	bool isMXSet           = false;
	bool isMYSet           = false;
	bool isDeltaSet        = false;
	bool isASet            = false;
	bool isDxSet           = false;
	bool isDtSet           = false;
	bool isNbrTStepSet     = false;
	bool isTStepsByDumpSet = false;
	bool isNoiseSet		   = false;
	bool isDeltaMuSet	   = false;
	bool isOmegaSet        = false;
	bool isAlphaSet        = false;
	bool isBSet			   = false;
	bool isInitMethodSet   = false;
	bool isModelTypeSet        = false;
	bool isRessourceUsedSet    = false;
	bool isPrecisionTypeSet    = false;
	bool isLineTreated     = false;
	while(std::getline(parameters_file, line)) {
		// Remove useless spaces
		line.erase(remove_if(line.begin(), line.end(), isspace), line.end());
		// Convert to lowercase in order to lighten line tests
		std::transform(line.begin(), line.end(), line.begin(), ::tolower);
		// Test if current line is a comment or empty (ignore if so)
		if (line[0] != '#' && line.length() > 0) {
			// Test if current line is correctly formatted (raise error if so)
			if (isParameterLineCorrect(line) != true) {
				std::cout << "Line " << lineNbr << " is not correctly formated : " << line << std::endl;
				exit(EXIT_FAILURE);
			}
			// Reset local variables for processing the line
			memset(parameter_str, '\0', STRING_LENGTH);
			value = 0.0;
			// Processing current line, waiting 2 items : parameter string and its value
			if (sscanf(line.c_str(), "%[^=]=%lf", parameter_str, &value) < 2) {
				// For debug purpose
				//std::cout << "Input line : " << line << ", parameter string : " << parameter_str << ", value : " << value << std::endl;
				// Error while processing a line, exiting program
				std::cout << "Error while reading line " << lineNbr << " : " << line << std::endl;
				exit(EXIT_FAILURE);
			} else {
				isLineTreated = false;
				// The comparison is NOT case sensitive (lowercase conversion done upper)
				if (strcmp(parameter_str, "mx") == 0) {
					if (isMXSet == true) {
						std::cout << "Box's size in X (MX) is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->MX = (uint64_t)floor(value);
					isMXSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "my") == 0) {
					if (isMYSet == true) {
						std::cout << "Box's size in Y (MY) is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->MY = (uint64_t)floor(value);
					isMYSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "delta") == 0) {
					if (isDeltaSet == true) {
						std::cout << "Interface thickness (delta) is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->delta = value;
					isDeltaSet = true;
					isLineTreated = true;
				} else  if (strcmp(parameter_str, "a") == 0) {
					if (isASet == true) {
						std::cout << "Constant a is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->a = value;
					isASet = true;
					isLineTreated = true;
				} else  if (strcmp(parameter_str, "dx") == 0) {
					if (isDxSet == true) {
						std::cout << "Delta-X (dx) is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->dx = value;
					isDxSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "dt") == 0) {
					if (isDtSet == true) {
						std::cout << "Delta-T (dt) is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->dt= value;
					isDtSet = true;
					isLineTreated = true;
				} else  if (strcmp(parameter_str, "nbrtstep") == 0) {
					if (isNbrTStepSet == true) {
						std::cout << "Number of time steps (nbrTStep) is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->nbrTStep = (uint64_t)floor(value);
					isNbrTStepSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "tstepsbydump") == 0) {
					if (isTStepsByDumpSet == true) {
						std::cout << "Number of time steps (nbrTStep) is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->TStepsByDump = (uint64_t)floor(value);
					isTStepsByDumpSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "noise") == 0) {
					if (isNoiseSet == true) {
						std::cout << "Constant Noise is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->noise = value;
					isNoiseSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "deltamu") == 0) {
					if (isDeltaMuSet == true) {
						std::cout << "Delta Mu is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->deltamu = value;
					isDeltaMuSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "omega") == 0) {
					if (isOmegaSet == true) {
						std::cout << "Constant Omega is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->omega = value;
					isOmegaSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "alpha") == 0) {
					if (isAlphaSet == true) {
						std::cout << "Constant Alpha is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->alpha = value;
					isAlphaSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "b") == 0) {
					if (isBSet == true) {
						std::cout << "Constant b is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->b = value;
					isBSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "init") == 0) {
					if (isInitMethodSet == true) {
						std::cout << "Initialization method is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					p->initMethod = (int)floor(value) + INIT_METHOD_BASE;
					isInitMethodSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "model") == 0) {
					if (isModelTypeSet == true) {
						std::cout << "Model type is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					// Change model type only if the calling program has not already set it
					if (p->modelType == 0) {
						p->modelType = (int)floor(value) + MODEL_TYPE_BASE;
					}
					isModelTypeSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "ressource") == 0) {
					if (isRessourceUsedSet == true) {
						std::cout << "Ressource to use is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					// Change the ressource to use only if the calling program has not already set it
					if (p->resourceUsed == 0) {
						p->resourceUsed = (int)floor(value) + RESSOURCE_USED_BASE;
					}
					isRessourceUsedSet = true;
					isLineTreated = true;
				} else if (strcmp(parameter_str, "precision") == 0) {
					if (isPrecisionTypeSet == true) {
						std::cout << "Precision type is already set, error while processing line " << lineNbr << " : " << line << std::endl;
						exit(EXIT_FAILURE);
					}
					// Change precision type only if the calling program has not already set it
					if (p->precisionType == 0) {
						p->precisionType = (int)floor(value) + PRECISION_TYPE_BASE;
					}
					isPrecisionTypeSet = true;
					isLineTreated = true;
				}
				if (isLineTreated == false) {
					// Error while processing a line, exiting program
					std::cout << "Error while processing line " << lineNbr << " : " << line << std::endl;
					exit(EXIT_FAILURE);
				}
			}
		}
		lineNbr++;
	}
	parameters_file.close();
	// Ignoring isModelTypeSet, isRessourceUsedSet, isPrecisionTypeSet because there is an executable for each target (CPU, GPU)
	//   where those paremeters are set manualy.
	// Test if all parameters are set (simple model)
	if (p->modelType == MODEL_TYPE_SIMPLE) {
		if (isMXSet == false || isMYSet == false || isDeltaSet    == false || isASet            == false ||
			isDxSet == false || isDtSet == false || isNbrTStepSet == false || isTStepsByDumpSet == false ||
			isInitMethodSet  == false) {
			std::cout << "Error : some parameters are missing for the simulation" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	// Test if all parameters are set (complete model)
    else {
		if (isMXSet == false || isMYSet    == false || isDeltaSet    == false || isASet            == false ||
			isDxSet == false || isDtSet    == false || isNbrTStepSet == false || isTStepsByDumpSet == false ||
            isBSet  == false || isAlphaSet == false || isOmegaSet    == false || isDeltaMuSet      == false ||
			isNoiseSet == false || isInitMethodSet  == false) {
			std::cout << "Error : some parameters are missing for the simulation" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
    return(EXIT_SUCCESS);
}

/*
 * Initialize a simulationParameters_t struct passed by parameter with zeros,
 *   to ensure that first test on inner values will compare with 0.
 */
int initSimulationParameters (simulationParameters_t *sp)
{
	sp->MX = 0;
	sp->MY = 0;
	sp->delta = 0.;
	sp->a = 0.;
	sp->dx = 0.;
	sp->dt = 0.;
	sp->nbrTStep = 0;
	sp->TStepsByDump = 0;
	sp->noise = 0.;
	sp->deltamu = 0.;
	sp->omega = 0.;
	sp->alpha = 0.;
	sp->b = 0.;
	sp->initMethod = 0;
	sp->modelType = 0;
	sp->resourceUsed = 0;
	sp->precisionType = 0;
	return(EXIT_SUCCESS);
}

/*
 * Initialize a simulationData_t struct passed by parameter with zeros or NULL for pointers,
 *   to ensure that first test on inner values will compare with 0.
 */
int initSimulationData (simulationData_t *sd)
{
	sd->GHOST_POINTS = 0;
	sd->MXG = 0;
	sd->MXGH = 0;
	sd->MYG = 0;
	sd->MYGH = 0;
	sd->u1.reset();
	sd->u2.reset();
	sd->us = NULL;
	sd->ud = NULL;
	sd->factor1 = 0.;
	sd->factor2 = 0.;
	sd->invdxp2m12 = 0.;
	sd->theta.reset();
	sd->w.reset();
	sd->dxddelta = 0.;
	sd->invdxm2 = 0.;
	sd->deltap2 = 0.;
	sd->dtddeltap2 = 0.;
	sd->factor3 = 0.;
	sd->factor4 = 0.;
	sd->noisem2 = 0.;
	return(initSimulationParameters(&sd->p));
}

/*
 * Initialize data needed to run the simulation.
 * Check if the memory allocation of arrays are ok.
 * For optimization purpose, some terms are pre-calculated.
 */
int initData (simulationData_t *d)
{
    if (d->p.modelType == MODEL_TYPE_SIMPLE) {
		d->GHOST_POINTS = GHOST_POINTS_CPU_SIMPLE;
    } else {
		// Complete model
		if (d->p.resourceUsed == RESSOURCE_USED_GPU) {
			d->GHOST_POINTS = GHOST_POINTS_GPU_COMPLETE;
		} else {
    		d->GHOST_POINTS = GHOST_POINTS_CPU_COMPLETE;
		}
    }
    d->MXG  = d->p.MX + (2 * d->GHOST_POINTS);
    d->MXGH = d->p.MX +      d->GHOST_POINTS;
    d->MYG  = d->p.MY + (2 * d->GHOST_POINTS);
    d->MYGH = d->p.MY +      d->GHOST_POINTS;
	// Memory allocations for the simple model
    d->u1.reAllocate(d->MXG, d->MYG);
    if (d->u1.isArraySetup() == true) {
		d->u1.contentInitialization(0.0);
		d->u2.reAllocate(d->MXG, d->MYG);
		if (d->u2.isArraySetup() == true) {
			d->u2.contentInitialization(0.0);
			d->us = &d->u1;
			d->ud = &d->u2;
			// Optimizations for the simple model
			d->factor1 = d->p.dt * d->p.a / 2.0;
			d->factor2 = -2.0 / (d->p.delta * d->p.delta);
			d->invdxp2m12 = 1. / (12. * d->p.dx * d->p.dx);
			// Memory allocations for the complete model
			if (d->p.modelType == MODEL_TYPE_COMPLETE) {
				d->theta.reAllocate(d->MXG, d->MYG);
				if (d->theta.isArraySetup() == true) {
					d->theta.contentInitialization(0.0);
					d->w.reAllocate(d->MXG, d->MYG);
					if (d->w.isArraySetup() == true) {
						d->w.contentInitialization(0.0);
						// Optimizations for the complete model, precalculated terms
						d->dxddelta   = d->p.dx / d->p.delta;
						d->invdxm2    = 1. / (2. * d->p.dx);
						d->deltap2    = pow(d->p.delta, 2.);
						d->dtddeltap2 = d->p.dt * (1. / pow(d->p.delta, 2.));
						d->factor3    = d->p.dt * (pow(d->p.alpha, 2.) / pow(d->p.delta, 4.));
						d->factor4    = d->p.dt / (d->p.delta * d->p.b);
						d->noisem2    = 2. * d->p.noise;
						// End of initData() for the complete model
						return(EXIT_SUCCESS);
					} else {
						std::cout << "Error while allocating array w for the simulation" << std::endl;
						resetData(d);
						return(EXIT_FAILURE);
					}
				} else {
					std::cout << "Error while allocating array theta for the simulation" << std::endl;
					resetData(d);
					return(EXIT_FAILURE);
				}
			} else {
				// End of initData() for the simple model
				return(EXIT_SUCCESS);
			}
		} else {
			std::cout << "Error while allocating array u2 for the simulation" << std::endl;
			resetData(d);
			return(EXIT_FAILURE);
		}
    } else {
        std::cout << "Error while allocating array u1 for the simulation" << std::endl;
	    resetData(d);
	    return(EXIT_FAILURE);
    }
}

/*
 * Desallocation arrays needed after the simulation has finished.
 */
int resetData (simulationData_t *d)
{
	d->u1.reset();
	d->u2.reset();
     if (d->p.modelType == MODEL_TYPE_COMPLETE) {
		d->theta.reset();
		d->w.reset();
     }
    return(EXIT_SUCCESS);
}

/*
 * Dump simulation's data (array u) to a text file and specify the time stamp used.
 * For the complete model, array w and theta are also written.
 */
int writeData (simulationData_t *d, uint64_t timeStamp)
{
    // filename preparation
    std::stringstream filename;
    filename << "u_" << timeStamp;
    // Opening of file
    std::ofstream dump_file;
    dump_file.open(filename.str(), std::ofstream::out | std::ofstream::trunc);
    if (dump_file.is_open() != true) {
		std::cout << "Error opening file : " << filename.str() << std::endl;
		resetData(d);
		exit(EXIT_FAILURE);
	}
	// Writing U array
	for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
		for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) dump_file << d->us->at(i,j) << " ";
		dump_file << std::endl;
	}
	dump_file.close();
	// Writing theta and w array for the complete model only
    if (d->p.modelType == MODEL_TYPE_COMPLETE) {
	    std::stringstream filename;
		//// theta array
	    filename << "theta_" << timeStamp;
	    // Opening of file
	    std::ofstream dump_file;
	    dump_file.open(filename.str(), std::ofstream::out | std::ofstream::trunc);
		if (dump_file.is_open() != true) {
			std::cout << "Error opening file : " << filename.str() << std::endl;
			resetData(d);
			exit(EXIT_FAILURE);
		}
		// Writing theta array
		for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
			for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) dump_file << d->theta.at(i, j) << " ";
			dump_file << std::endl;
		}
		dump_file.close();
		// Reset the string stream
		filename.str("");
		//// w array
	    filename << "w_" << timeStamp;
	    // Opening of file
	    dump_file.open(filename.str(), std::ofstream::out | std::ofstream::trunc);
		if (dump_file.is_open() != true) {
			std::cout << "Error opening file : " << filename.str() << std::endl;
			resetData(d);
			exit(EXIT_FAILURE);
		}
		// Writing w array
		for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
			for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) dump_file << d->w.at(i, j) << " ";
			dump_file << std::endl;
		}
		dump_file.close();
	}
    return(EXIT_SUCCESS);
}

/*
 * Swap pointers between U vector source and destination to avoid losing time while
 *   copying data at the end of each calculated time step
 */
int swapArrayPointers (simulationData_t *d)
{
	CArray2D<double> *tmp;
	tmp = d->us;
	d->us = d->ud;
	d->ud = tmp;
	return(EXIT_SUCCESS);
}

/*
 * Swap pointers between 2 vectors of float
 */
int swapArrayPointersFloat (float **a, float **b)
{
	float *tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
	return(EXIT_SUCCESS);
}

/*
 * Swap pointers between 2 vectors of double
 */
int swapArrayPointersDouble (double **a, double **b)
{
	double *tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
	return(EXIT_SUCCESS);
}
