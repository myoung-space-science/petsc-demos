petscdir ?= ~/petsc

include ${petscdir}/lib/petsc/conf/variables
include ${petscdir}/lib/petsc/conf/rules

dmswarm01: dmswarm01.o
	-${CLINKER} -o dmswarm01 dmswarm01.o ${PETSC_LIB}
	${RM} dmswarm01.o

dmswarm-coords: dmswarm-coords.o
	-${CLINKER} -o dmswarm-coords dmswarm-coords.o ${PETSC_LIB}
	${RM} dmswarm-coords.o

sobol-migrate: sobol-migrate.o
	-${CLINKER} -o sobol-migrate sobol-migrate.o ${PETSC_LIB}
	${RM} sobol-migrate.o
