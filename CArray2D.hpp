/************************************************
* Monguillon Julien                             *
* L.PMC - CNRS / Ecole Polytechnique            *
************************************************/

#include <sys/types.h>					// GNU C Library - POSIX Standard: 2.6 Primitive System Data Types
#include <cstdint>						// uint64_t
#include <cstdlib>					    // EXIT_SUCCESS & EXIT_FAILURE

#include <new>							// std::nothrow
#include <iostream>                     // std::cout

template <typename T>
class CArray2D
{
	/*************************************************************************/
	/*** private data members                                              ***/
	/*************************************************************************/
	private:
		uint64_t m_nRows;
		uint64_t m_nColumns;
		uint64_t m_nCells;
		bool     m_IsArraySetup;
		T*       m_Array;
	
	/*************************************************************************/
	/*** public methods                                                    ***/
	/*************************************************************************/
	public:
		// Constructor with no parameter
		CArray2D(void)
		{
			m_nRows    = 0;
			m_nColumns = 0;
			m_nCells   = 0;
			m_IsArraySetup = false;
			m_Array = NULL;
		}

		// Constructor with rows and columns number specified
		CArray2D(uint64_t t_nRows, uint64_t t_nColumns)
		{
			m_Array = NULL;
			m_IsArraySetup = false;
			if(t_nRows == 0 || t_nColumns == 0) {
                std::cout << "Error : rows and columns number must be superior to 0" << std::endl;
			} else {
                reAllocate(t_nRows, t_nColumns);
			}
		}

		// Destructor : just free the array if allocated
		~CArray2D()
		{
			reset();
		}

		// Reset object
		void reset (void) {
			if (m_IsArraySetup == true) {
				if(m_Array != NULL) delete[] m_Array;
				m_nRows    = 0;
				m_nColumns = 0;
				m_nCells   = 0;
				m_IsArraySetup = false;
			}
		}

		// Re-allocation of 2D array
		int reAllocate(uint64_t t_nRows, uint64_t t_nColumns)
		{
			reset();
			if(t_nRows == 0 || t_nColumns == 0) {
                std::cout << "Error : row and columns number must be superior to 0" << std::endl;
                return EXIT_FAILURE;
			}
			m_nRows    = t_nRows;
			m_nColumns = t_nColumns;
			m_nCells   = t_nRows * t_nColumns;
			m_Array    = new (std::nothrow) T[m_nCells];
			if(m_Array == NULL) {
                std::cout << "Error : memory allocation for the array" << std::endl;
				return EXIT_FAILURE;
			} else {
				m_IsArraySetup = true;
				return EXIT_SUCCESS;
			}
		}

		// Set all cells to the value passed as parameter
		int contentInitialization(T value)
		{
			if(m_IsArraySetup != true || m_Array == NULL) {
                std::cout << "Error : you are trying to initialize an array not setup" << std::endl;
                return EXIT_FAILURE;
			}
			for(uint64_t i = 0; i < m_nCells; ++i) m_Array[i] = value;
			return EXIT_SUCCESS;
		}

		/****************************** copy methods ******************************/

		// Manual copy from an CArray2D passed by parameter. otherArray was created with automatic storage duration
		int copyFrom(CArray2D<T> &otherArray) {
			if (otherArray.isArraySetup() != true || m_nRows > otherArray.getRows() || m_nColumns > otherArray.getColumns()) {
				std::cout << "Error : arrays must have the same size to be copied, or the source must be greater" << std::endl;
				return EXIT_FAILURE;
			}
			for(uint64_t i = 0; i < m_nCells; ++i) m_Array[i] = otherArray[i];
			return EXIT_SUCCESS;
		}

		// Manual copy from an CArray2D passed by parameter. otherArray was created with dynamic storage duration (new/delete)
		int copyFrom(CArray2D<T>* &otherArray) {
			if (otherArray->isArraySetup() != true || m_nRows > otherArray->getRows() || m_nColumns > otherArray->getColumns()) {
				std::cout << "Error : arrays must have the same size to be copied, or the source must be greater" << std::endl;
				return EXIT_FAILURE;
			}
			for(uint64_t i = 0; i < m_nCells; ++i) m_Array[i] = (*otherArray)[i];
			return EXIT_SUCCESS;
		}

		// To prevent unwanted copying
		CArray2D(const CArray2D<T>&);
		CArray2D& operator = (const CArray2D<T>&);

		/****************************** getter methods ******************************/

		// get private attributes
		uint64_t getRows()      const { return m_nRows;        }
		uint64_t getColumns()   const { return m_nColumns;     }
		uint64_t getCells()     const { return m_nCells;       }
		bool     isArraySetup() const { return m_IsArraySetup; }
		T*       getArray()			  { return m_Array;        }

		// Indexing (parenthesis operator), two of them (for const correctness)
		const T& operator () (uint64_t iRow, uint64_t iColumn) const {
			return m_Array[iColumn * m_nRows + iRow];
		}
		T& operator () (uint64_t iRow, uint64_t iColumn) {
			return m_Array[iColumn * m_nRows + iRow];
		}

		// Accessor at() more secure because it check if coordinate are not out of range
		T at(uint64_t iRow, uint64_t iColumn) {
			if (iRow < m_nRows && iColumn < m_nColumns) {
				return m_Array[iColumn * m_nRows + iRow];
			} else {
				std::cout << "Error : given coordinates are out of range for read" << std::endl;
				return 0;
			}
		}

		T at(uint64_t x) {
			if (x < m_nCells) {
				return m_Array[x];
			} else {
				std::cout << "Error : given coordinates are out of range for read" << std::endl;
				return 0;
			}
		}

        // Indexing (bracket operator), two of them (for const correctness)
        const T& operator [] (uint64_t x) const {
            return m_Array[x];
		}
		
		T& operator [] (uint64_t x) {
            return m_Array[x];
        }

		void printToConsole () {
			for(int i = 0; i < m_nRows; ++i) {
				for(int j = 0; j < m_nRows; ++j) {
					std::cout << m_Array[j * m_nRows + i] << " ";
				}
				std::cout << std::endl;
			}
		}

		/****************************** setter methods ******************************/

		// Change the value of a single cell in the array, but check given coordinates
		int set(uint64_t iRow, uint64_t iColumn, T value) {
			if (iRow >= m_nRows || iColumn >= m_nColumns) {
				std::cout << "Error : given coordinates are out of range for write" << std::endl;
				return EXIT_FAILURE;
			}
			m_Array[iColumn * m_nRows + iRow] = value;
			return EXIT_SUCCESS;
		}
};
