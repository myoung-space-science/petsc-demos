petscdir ?= ~/petsc

include ${petscdir}/lib/petsc/conf/variables
include ${petscdir}/lib/petsc/conf/rules

dmswarm01: dmswarm01.o
	-${CLINKER} -o dmswarm01 dmswarm01.o ${PETSC_LIB}
	${RM} dmswarm01.o

