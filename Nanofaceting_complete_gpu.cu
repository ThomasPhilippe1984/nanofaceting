/************************************************
* Philippe Thomas								*
* Monguillon Julien								*
* L.PMC - CNRS / Ecole Polytechnique			*
************************************************/

// Project library
#include "Nanofaceting.h"

/* Standard CUDA library */
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include "helper_cuda.h"			// Utility library from CUDA samples

/*** Data structure **********************************************************/

// Data structure to easy calculation on GPU, with pre-calculated terms
typedef struct
{
	// System size
	uint64_t MX;					// Box's size in X
	uint64_t MY;					// Box's size in Y
	// Simulation precision
	float dx;						// delta-X
	float dt;						// delta-T
	// Constants
	float delta;					// Interface thickness
	float a;						// Gamma function's parameter
	float noise;					// Noise amplitude (thermal fluctuation)
	float deltamu;					// Driving force for crystallization
	float omega;					// Gamma function's parameter
	float alpha;					// Corner size
	float b;						// Constant in the term for driving force for growth (if growth activated)
	// Pre-calculated terms
	float deltap2;
	float invdxp2m12;
	float invdxm2;
	float dtddeltap2;
	float factor3;
	float factor4;
	float noisem2;
} simulationParametersGPU_t;

/*** Function declarations ***************************************************/

int computeSimulation			(simulationData_t*);
int initSimulationGPU			(simulationData_t*, float*);
int initSimulationParametersGPU	(simulationParametersGPU_t*, simulationParameters_t);
int writeDataGPU				(simulationData_t*, uint64_t, float*, float*, float*);

__global__ void	updateModel (float*, float*, float*, simulationParametersGPU_t);
__global__ void	translate	(float*, float*, simulationParametersGPU_t);
__global__ void	front(float*, float*, simulationParametersGPU_t);
__global__ void	orientation(float*, float*, simulationParametersGPU_t);

/*** Code ********************************************************************/

// Main function
int main (int argNbr, char *argStr[])
{
	simulationData_t simulationData;
	initSimulationData(&simulationData);
	// Forcing some options
	simulationData.p.modelType     = MODEL_TYPE_COMPLETE;
	simulationData.p.resourceUsed  = RESSOURCE_USED_GPU;
	simulationData.p.precisionType = PRECISION_TYPE_SP;
    // Parsing parameters file
    if (parseParameters(&simulationData.p, argNbr, argStr) != EXIT_SUCCESS) {
        exit(EXIT_FAILURE);
    }
    printParameters(simulationData.p);
    // Lauching simulation
    if (initData(&simulationData) == EXIT_SUCCESS) {
        if (computeSimulation(&simulationData) != EXIT_SUCCESS) {
            std::cout << "Error during computation" << std::endl;
            resetData(&simulationData);
            exit(EXIT_FAILURE);
        }
    } else {
		exit(EXIT_FAILURE);
    }
    resetData(&simulationData);
	system("PAUSE");
    return(EXIT_SUCCESS);
}

/*
 * Manage main loop on time
 */
 int computeSimulation (simulationData_t* d)
{
	// Variables for time measurement
	clock_t start, finish;
	double duration;
	start = clock();
	// Initialize random seed
	srand(1324ULL);
	// Prepare for compute
	time_t rawtime;
	struct tm * timeinfo;
	size_t totalSize = d->p.MX * d->p.MY * sizeof(float);
	size_t halfSize = d->p.MX * sizeof(float);
	// Local memory management on host (CPU), the dedicated 2D array class is not usuable with CUDA functions
	float *h_u  = NULL;
	h_u = (float*)malloc(totalSize);
	if (h_u == NULL) {
		std::cout << "Error while allocating h_u" << std::endl;
		resetData(d);
		return(EXIT_FAILURE);
	}
	float *h_y = NULL;
	h_y = (float*)malloc(halfSize);
	if (h_y == NULL) {
		std::cout << "Error while allocating h_y" << std::endl;
		resetData(d);
		return(EXIT_FAILURE);
	}
	float *h_theta = NULL;
	h_theta = (float*)malloc(halfSize);
	if (h_theta == NULL) {
		std::cout << "Error while allocating h_theta" << std::endl;
		resetData(d);
		return(EXIT_FAILURE);
	}
	initSimulationGPU(d, h_u);
	simulationParametersGPU_t spGPU;
	initSimulationParametersGPU(&spGPU, d->p);
	// CUDA management - start
	checkCudaErrors( cudaSetDevice(gpuDeviceInit(0)) );
	int target = 2;
	float *d_u1 = NULL;
	float *d_u2 = NULL;
	float *d_ur = NULL;											// Randomly generated numbers, same size than the system
	float *d_y = NULL;
	float *d_theta = NULL;
	checkCudaErrors( cudaMalloc( (void **)&d_u1, totalSize) );
	checkCudaErrors( cudaMalloc( (void **)&d_u2, totalSize) );
	checkCudaErrors( cudaMalloc( (void **)&d_ur, totalSize) );
	checkCudaErrors( cudaMalloc((void **)&d_y, halfSize)    );
	checkCudaErrors( cudaMalloc((void **)&d_theta, halfSize));
	if (d_u1 == NULL || d_u2 == NULL || d_ur == NULL || d_y == NULL || d_theta == NULL) {
		std::cout << "Error while allocating memory on the GPU" << std::endl;
		resetData(d);
		free(h_u);
		free(h_y);
		free(h_theta);
		return(EXIT_FAILURE);
	}
	checkCudaErrors( cudaMemcpy(d_u1, h_u, totalSize, cudaMemcpyHostToDevice) );
	dim3 block(8, 8, 1);
	dim3 block2(8, 1, 1);
	dim3 grid(cuda_iDivUp(d->p.MX, block.x), cuda_iDivUp(d->p.MY, block.y), 1);
	dim3 grid2(cuda_iDivUp(d->p.MX, block.x), 1, 1);
	curandGenerator_t curandGenerator;
//	checkCudaErrors( curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT) );
//	checkCudaErrors( curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL) );
	// Dump data at initial conditions
	writeDataGPU(d, 0, h_u, h_y, h_theta);
	// Initializing the integer k0 so that to compute the front's shift (when normal growth) 
	uint64_t k0 = 0;
	// Main time loop
	for(uint64_t k = 1; k <= d->p.nbrTStep; ++k) {
		// New bunch of randomly generated numbers, more efficient than generate a new one in every thread
//		checkCudaErrors( curandGenerateUniform(curandGenerator, d_ur, totalSize) );
		if (target == 2) {
			updateModel<<<grid, block>>>(d_u1, d_u2, d_ur, spGPU);
			target = 1;
		} else {
			updateModel<<<grid, block>>>(d_u2, d_u1, d_ur, spGPU);
			target = 2;
		}
		// Translating the front of dx when necessary (a coefficient is introduced to account for the surface slope and may vary) 
		k0 = k0 + 1;
		if (k0 * d->p.dt * 1.27 >= d->p.dx) {
			// translating the last calculated u on GPU
			if (target == 2) {
				translate <<<grid, block >>>(d_u1, d_u2, spGPU);
				target = 1;
			}
			else {
				translate <<<grid, block >>>(d_u2, d_u1, spGPU);
				target = 2;
			}
			k0 = 0;
		}
		// Dump data ?
		if (k % d->p.TStepsByDump == 0) {
			// Wait for CUDA kernel to finish
			checkCudaErrors( cudaDeviceSynchronize() );
			// Computing and copying last calculated u and the front position y on GPU to host (CPU)
			if (target == 2) {
				front <<<grid2, block2 >>>(d_u1, d_y, spGPU);
				checkCudaErrors( cudaMemcpy(h_u, d_u1, totalSize, cudaMemcpyDeviceToHost) );
				checkCudaErrors(cudaMemcpy(h_y, d_y, halfSize, cudaMemcpyDeviceToHost));
			} else {
				front <<<grid2, block2 >>>(d_u2, d_y, spGPU);
				checkCudaErrors( cudaMemcpy(h_u, d_u2, totalSize, cudaMemcpyDeviceToHost) );
				checkCudaErrors(cudaMemcpy(h_y, d_y, halfSize, cudaMemcpyDeviceToHost));
			}
			// Computing and copying orientation
			orientation <<<grid2, block2 >>>(d_y, d_theta, spGPU);
			checkCudaErrors(cudaMemcpy(h_theta, d_theta, halfSize, cudaMemcpyDeviceToHost));
			// Dump to file
			time(&rawtime);
			timeinfo = localtime(&rawtime);
			std::cout << "Data dump at time step number " << k << "/" << d->p.nbrTStep << " - " << asctime(timeinfo);
			writeDataGPU(d, k, h_u, h_y, h_theta);
		}
	}
	// CUDA management - end
	checkCudaErrors( cudaFree(d_u1) );
	checkCudaErrors( cudaFree(d_u2) );
	checkCudaErrors( cudaFree(d_ur) );
	checkCudaErrors( cudaFree(d_y)  );
	checkCudaErrors( cudaFree(d_theta));
	cudaDeviceReset();
	// Time spent calculation
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << "Simulation finished in " << duration << " seconds" << std::endl;
	return(EXIT_SUCCESS);
 }
 
 /*
 * Initial conditions, with local CPU (host) vector
 */
int initSimulationGPU (simulationData_t* d, float *u)
{
	// TODO : Type de RNG -> seed different a gerer, uniforme, petite periode
	uint64_t MYGb2 = d->MYG / 2;
	uint64_t MXGb2 = d->MXG / 2;
	// Planar front, noisy and smooth with tanh profiles, to avoid numerical instabilities at first time steps
	if (d->p.initMethod == INIT_METHOD_PLAN_SMOOTH) {
		for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) {
			for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
			u[i + j*d->p.MX] = -tanh( (j * 1. - ((MYGb2/2 + rand() % 6) * 1.)) * d->dxddelta);
			}
		}
	} else if (d->p.initMethod == INIT_METHOD_PLAN_ABRUPT) {
		// Planar front, noisy but abrupt
		for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) {
			for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
				if (j > (MYGb2 + rand() % 10 ) ) {
					u[i + j*d->p.MX] = -1.;
				}
				else {
					u[i + j*d->p.MX] = 1.;
				}
			}
		}
	} else if (d->p.initMethod == INIT_METHOD_SPHERE) {
		// Spherical interface, manual size for benchmark
		for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) {
			for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
				if ((j - MYGb2)*(j- MYGb2) + (i - MXGb2)*(i - MXGb2) > 16*16) {
					u[i + j*d->p.MX] = -1.;
				}
				else {
					u[i + j*d->p.MX] = 1.;
				}
			}
		}
	} else {
		// Default initialization, filled with zeros
		for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) {
			for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
				u[i + j*d->p.MX] = 0.;
			}
		}
	}
	return(EXIT_SUCCESS);
}

/*
 * Initialization of the struct passed to cuda kernel.
 * It contains the system size, Simulation precision,
 *   float conversions and pre-calculated terms.
 */
int initSimulationParametersGPU	(simulationParametersGPU_t *spGPU, simulationParameters_t sp)
{
	// System size
	spGPU->MX = sp.MX;
	spGPU->MY = sp.MY;
	// Simulation precision and length
	spGPU->dx = (float)sp.dx;
	spGPU->dt = (float)sp.dt;
	// Float conversion
	spGPU->delta   = (float)sp.delta;
	spGPU->a       = (float)sp.a;
	spGPU->noise   = (float)sp.noise;
	spGPU->deltamu = (float)sp.deltamu;
	spGPU->omega   = (float)sp.omega;
	spGPU->alpha   = (float)sp.alpha;
	spGPU->b       = (float)sp.b;
	// Pre-calculated terms
	spGPU->deltap2    = (float)(pow(sp.delta, 2.));
	spGPU->invdxp2m12 = (float)(1. / (12. * pow(sp.dx, 2.)));
	spGPU->invdxm2    = (float)(1. / (2. * sp.dx));
	spGPU->dtddeltap2 = (float)(sp.dt * (1. / pow(sp.delta, 2.)));
	spGPU->factor3    = (float)(sp.dt * (pow(sp.alpha, 2.) / pow(sp.delta, 4.)));
	spGPU->factor4    = (float)(sp.dt / (sp.delta * sp.b));
	spGPU->noisem2    = (float)(2. * sp.noise);
	return(EXIT_SUCCESS);
}
/*
*Translating the front
*/
__global__ void translate(float *us, float *ud, simulationParametersGPU_t p)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int jj = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ii + jj * p.MX;
	// new value of the field
	if (jj < p.MY) {
		ud[idx] = us[ii + (jj + 1) * p.MX];
	}
	else {
		ud[idx] = us[idx];
	}
}
/*
*Computing the front position y
*/
__global__ void front(float *us, float *y, simulationParametersGPU_t p)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = ii;

	y[idx] = 0.f;

	//y
	for (int jj = 0; jj < p.MY; jj++)
	{
		if (us[jj * p.MX + ii] > 0.f & us[(jj+1) * p.MX + ii] < 0.f)
		{
			float u0 = us[jj * p.MX + ii];
			float u1 = us[(jj + 1) * p.MX + ii];
			float slope = (u1 - u0) / p.dx;
			float deltay = -u0 / slope;
			y[idx] = jj * p.dx + deltay;
		}
	}
}
/*
*Computing the orientation of the front
*/
__global__ void orientation(float *y, float *theta, simulationParametersGPU_t p)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = ii;

	theta[idx] = 0.f;
	float h[5];	//represents theta

	//h
	for (int di = -2; di <= 2; di++) {
		// Remaining of modulo for periodic boundary conditions
		int i = (ii + di + p.MX) % p.MX;
		h[di + 2] = y[i];
	}
	//theta
	float q = (-h[4] +8.f*h[3]- 8.f*h[1] +h[0]) / (12*p.dx);
	theta[idx] = atan(q);
}
/*
 * Update the model at each time step
 */
__global__ void updateModel(float *us, float *ud, float *urand, simulationParametersGPU_t p)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ii + jj * p.MX;

	float y[9][9];	//represents u, loading some values of the field, y[4][4] is the value in i,j
	float z[5][5];	//represents the quantity w
	float x[3][3];	//represents theta
	
	// y
	for(int dj = -4; dj <= 4; dj++) {
		for(int di = -4; di <= 4; di++) {
			// Remaining of modulo for periodic boundary conditions
			int i = (ii + di + p.MX) % p.MX;
			int j = (jj + dj + p.MY) % p.MY;
			y[di + 4][dj + 4] = us[i + j*p.MX];
			// If fixed boundary conditions in j (to 1 or -1)
			int k = jj + dj;
			int kk = jj - 1 + dj;
			int kkk = jj + 1 + dj;
			int kkkk = jj + 2 + dj;
			int kkkkk = jj - 2 + dj;
			int kkkkkk = jj + 3 + dj;
			int kkkkkkk = jj - 3 + dj;
			if (k > p.MY || kk > p.MY ||  kkk > p.MY ||  kkkk > p.MY ||  kkkkk  > p.MY ||  kkkkkk > p.MY ||  kkkkkkk > p.MY) {
				y[di + 4][dj + 4] = -1.0f;
			}
			if (k < 0 || kk < 0 || kkk < 0 || kkkk < 0 || kkkkk < 0 || kkkkkk < 0 || kkkkkkk < 0) {
				y[di + 4][dj + 4] = 1.0f;
				//-1 also
				//y[di + 4][dj + 4] = -1.0f;
			}
		}
	}

	// z
	for(int kj = 0; kj <= 4; kj++) {
		for(int ki = 0; ki <= 4; ki++) {
			z[ki][kj] = 2.0f * y[ki + 2][kj + 2] * (y[ki + 2][kj + 2] * y[ki + 2][kj + 2] - 1.0f) - p.deltap2 *
				( (16.0f*y[ki + 1][kj +2] - 30.0f*y[ki + 2][kj + 2] + 16.0f*y[ki + 3][kj + 2] - y[ki + 4][kj + 2] - y[ki][kj + 2]) * p.invdxp2m12 
				+ (16.0f*y[ki + 2][kj +1] - 30.0f*y[ki + 2][kj + 2] + 16.0f*y[ki + 2][kj + 3] - y[ki + 2][kj + 4] - y[ki + 2][kj]) * p.invdxp2m12);
		}
	}
	
	// x
	for(int nj = 0; nj <= 2; nj++) {
		for(int ni = 0; ni <= 2; ni++) {
			float gradux = (y[ni + 4][nj + 3] - y[ni + 2][nj + 3]) * p.invdxm2;
			float graduy = (y[ni + 3][nj + 4] - y[ni + 3][nj + 2]) * p.invdxm2;
			// we avoid n_x / n_y to diverge
			if (abs(graduy) >= 0.1f) {
				x[ni][nj] = atan(-gradux / graduy);
			}
			else {
				x[ni][nj] = 0.0f;
			}
		}
	}

	float y2 = y[4][4] * y[4][4];
	// phi prime function
	float phi_prime  = 2.0f * y[4][4] * (y2 - 1.0f);
	// p prime function
	float p_prime    = 9.0f * (1. - y2);
	// phi second
	float phi_second = 6.0f * y2 - 2.0f;
	// laplacian of the field
	float lap_u_x = (16.0f*y[5][4] - 30.0f*y[4][4] + 16.0f*y[3][4] - y[2][4] - y[6][4]) * p.invdxp2m12;
	float lap_u_y = (16.0f*y[4][5] - 30.0f*y[4][4] + 16.0f*y[4][3] - y[4][2] - y[4][6]) * p.invdxp2m12;
	// laplacien of w
	float lap_w_x = (16.0f*z[3][2] - 30.0f*z[2][2] + 16.0f*z[1][2] - z[0][2] - z[4][2]) * p.invdxp2m12;
	float lap_w_y = (16.0f*z[2][3] - 30.0f*z[2][2] + 16.0f*z[2][1] - z[2][0] - z[2][4]) * p.invdxp2m12;
	// the gamma function
	float gamma            = 1.0f + p.a * float(cos(4.0f * (x[1][1] - p.omega)));
	float gamma_at_iplus1  = 1.0f + p.a * float(cos(4.0f * (x[2][1] - p.omega)));
	float gamma_at_jplus1  = 1.0f + p.a * float(cos(4.0f * (x[1][2] - p.omega)));
	float gamma_at_iminus1 = 1.0f + p.a * float(cos(4.0f * (x[0][1] - p.omega)));
	float gamma_at_jminus1 = 1.0f + p.a * float(cos(4.0f * (x[1][0] - p.omega)));
	// the gamma prime function
	float gamma_prime            = -4.0f * p.a * float(sin(4.0f * (x[1][1] - p.omega)));
	float gamma_prime_at_iplus1  = -4.0f * p.a * float(sin(4.0f * (x[2][1] - p.omega)));
	float gamma_prime_at_jplus1  = -4.0f * p.a * float(sin(4.0f * (x[1][2] - p.omega)));
	float gamma_prime_at_iminus1 = -4.0f * p.a * float(sin(4.0f * (x[0][1] - p.omega)));
	float gamma_prime_at_jminus1 = -4.0f * p.a * float(sin(4.0f * (x[1][0] - p.omega)));
	// first derivatives of u
	float grad_u_x            = (y[5][4] - y[3][4]) * p.invdxm2;
	float grad_u_x_at_jplus1  = (y[5][5] - y[3][5]) * p.invdxm2;
	float grad_u_x_at_jminus1 = (y[5][3] - y[3][3]) * p.invdxm2;
	float grad_u_y            = (y[4][5] - y[4][3]) * p.invdxm2;
	float grad_u_y_at_iplus1  = (y[5][5] - y[5][3]) * p.invdxm2;
	float grad_u_y_at_iminus1 = (y[3][5] - y[3][3]) * p.invdxm2;

	// new value of the field
	ud[idx] = y[4][4] - p.dtddeltap2 * (gamma * phi_prime - p.deltap2 * (gamma * (lap_u_x + lap_u_y)
		+ grad_u_x * ((gamma_at_iplus1 - gamma_at_iminus1) * p.invdxm2) + grad_u_y * ((gamma_at_jplus1 - gamma_at_jminus1) * p.invdxm2)
		- gamma_prime * ((grad_u_y_at_iplus1 - grad_u_y_at_iminus1) * p.invdxm2 - (grad_u_x_at_jplus1 - grad_u_x_at_jminus1) * p.invdxm2)
		- grad_u_y * ((gamma_prime_at_iplus1 - gamma_prime_at_iminus1) * p.invdxm2) + grad_u_x * ((gamma_prime_at_jplus1 - gamma_prime_at_jminus1) * p.invdxm2)))
		+ p.factor3 * (p.deltap2 * (lap_w_x + lap_w_y) - phi_second * z[2][2])
		// + driving force
		+ p.deltamu * p.factor4 * p_prime
		// + noise
		+ p.noisem2 * urand[idx] * (1. - y[4][4]) * (1. + y[4][4]);
}

/*
 * Dump simulation's data (array u) to a text file and specify the time stamp used.
 * For the complete model, array w and theta are also written.
 */
 int writeDataGPU (simulationData_t *d, uint64_t timeStamp, float *u, float *y, float *theta)
 {
	// filename preparation
	//u
	std::stringstream filename;
	filename << "u_" << timeStamp;
	// Opening of file
	std::ofstream dump_file;
	dump_file.open(filename.str(), std::ofstream::out | std::ofstream::trunc);
	if (dump_file.is_open() != true) {
		std::cout << "Error opening file : " << filename.str() << std::endl;
		resetData(d);
		free(u);
		exit(EXIT_FAILURE);
	}
	// Writing U array
	for(uint64_t j = d->GHOST_POINTS; j < (d->MYGH); ++j) {
		for(uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) {
			dump_file << u[i + j*d->p.MX] << " ";
		}
		dump_file << std::endl;
	}
	dump_file.close();
	//Y
	// Reset the string stream
	filename.str("");
	filename << "y_" << timeStamp;
	// Opening of file
	dump_file.open(filename.str(), std::ofstream::out | std::ofstream::trunc);
	if (dump_file.is_open() != true) {
		std::cout << "Error opening file : " << filename.str() << std::endl;
		resetData(d);
		free(y);
		exit(EXIT_FAILURE);
	}
	// Writing Y array
		for (uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) {
			dump_file << y[i] << " ";
		dump_file << std::endl;
	}
	dump_file.close();
	//THETA
	// Reset the string stream
	filename.str("");
	filename << "theta_" << timeStamp;
	// Opening of file
	dump_file.open(filename.str(), std::ofstream::out | std::ofstream::trunc);
	if (dump_file.is_open() != true) {
		std::cout << "Error opening file : " << filename.str() << std::endl;
		resetData(d);
		free(theta);
		exit(EXIT_FAILURE);
	}
	// Writing THETA array
	for (uint64_t i = d->GHOST_POINTS; i < (d->MXGH); ++i) {
		dump_file << theta[i] << " ";
		dump_file << std::endl;
	}
	dump_file.close();
	return(EXIT_SUCCESS);
}
