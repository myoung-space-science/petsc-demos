static char help[] = \
"Example of initializing DMSwarm coordinates from a Sobol' sequence.\n\n\
This program allows the user (via a CLI) to adjust the size of the global domain \n\
as well as the size of the subdomain over which to distribute particles. \n\
It accepts the following parameters \n\
-nx [int]   : The number of grid cells in the x dimension \n\
-ny [int]   : The number of grid cells in the y dimension \n\
-nz [int]   : The number of grid cells in the z dimension \n\
-Lx [float] : The length of the x dimension \n\
-Ly [float] : The length of the y dimension \n\
-Lz [float] : The length of the z dimension \n\
-x0 [float] : The lower bound of particle positions in the x dimension \n\
-y0 [float] : The lower bound of particle positions in the y dimension \n\
-z0 [float] : The lower bound of particle positions in the z dimension \n\
-x1 [float] : The upper bound of particle positions in the x dimension \n\
-y1 [float] : The upper bound of particle positions in the y dimension \n\
-z1 [float] : The upper bound of particle positions in the z dimension \n\
-np [int]   : The number of particles to place along every dimension \n\
--periodic  : Use periodic boundaries for the grid \n";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

#define NDIM 3

typedef struct {
  PetscInt       nx, ny, nz;
  PetscReal      Lx, Ly, Lz;
  PetscReal      x0, y0, z0;
  PetscReal      x1, y1, z1;
  DMBoundaryType bc;
} Grid;

typedef struct {
  PetscInt np;
} Particles;

typedef struct {
  Grid      grid;
  Particles particles;
} UserContext;


/* Read parameter values from the options database.

See help text for parameter descriptions.
*/
static PetscErrorCode
ProcessOptions(UserContext *options)
{
  PetscFunctionBeginUser;

  PetscInt  intArg;
  PetscReal realArg;
  PetscBool boolArg, found;

  options->grid.nx = 7;
  options->grid.ny = 7;
  options->grid.nz = 7;
  options->grid.Lx = 1.0;
  options->grid.Ly = 1.0;
  options->grid.Lz = 1.0;
  options->grid.x0 = 0.0;
  options->grid.y0 = 0.0;
  options->grid.z0 = 0.0;
  options->grid.x1 = options->grid.Lx;
  options->grid.y1 = options->grid.Ly;
  options->grid.z1 = options->grid.Lz;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &intArg, &found));
  if (found) {
    options->grid.nx = intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &intArg, &found));
  if (found) {
    options->grid.ny = intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz", &intArg, &found));
  if (found) {
    options->grid.nz = intArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Lx", &realArg, &found));
  if (found) {
    options->grid.Lx = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Ly", &realArg, &found));
  if (found) {
    options->grid.Ly = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Lz", &realArg, &found));
  if (found) {
    options->grid.Lz = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-x0", &realArg, &found));
  if (found) {
    options->grid.x0 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-y0", &realArg, &found));
  if (found) {
    options->grid.y0 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-z0", &realArg, &found));
  if (found) {
    options->grid.z0 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-x1", &realArg, &found));
  if (found) {
    options->grid.x1 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-y1", &realArg, &found));
  if (found) {
    options->grid.y1 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-z1", &realArg, &found));
  if (found) {
    options->grid.z1 = realArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-np", &intArg, &found));
  if (found) {
    options->particles.np = intArg;
  } else {
    options->particles.np = -1;
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "--periodic", &boolArg, &found));
  if (found) {
    options->grid.bc = DM_BOUNDARY_PERIODIC;
  } else {
    options->grid.bc = DM_BOUNDARY_GHOSTED;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateMeshDM(DM *mesh, UserContext *user)
{
  PetscInt       nx=user->grid.nx;
  PetscInt       ny=user->grid.ny;
  PetscInt       nz=user->grid.nz;
  DMBoundaryType xBC=user->grid.bc;
  DMBoundaryType yBC=user->grid.bc;
  DMBoundaryType zBC=user->grid.bc;
  PetscInt       dof=2;
  PetscInt       width=1;

  PetscFunctionBeginUser;

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, width, NULL, NULL, NULL, mesh));
  PetscCall(DMDASetElementType(*mesh, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*mesh));
  PetscCall(DMSetUp(*mesh));
  PetscCall(DMDASetUniformCoordinates(*mesh, 0.0, user->grid.Lx, 0.0, user->grid.Ly, 0.0, user->grid.Lz));
  PetscCall(DMSetApplicationContext(*mesh, user));
  PetscCall(DMView(*mesh, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ViewSwarm(DM swarm, const char *filestem, UserContext user)
{
  char        binname[PETSC_MAX_PATH_LEN]="";
  char        xmfname[PETSC_MAX_PATH_LEN]="";
  PetscInt    npG, npL;
  PetscViewer viewer;
  Vec         target;
  int         rank;

  PetscFunctionBeginUser;

  // Build the binary file name.
  PetscCall(PetscStrcat(binname, filestem));
  PetscCall(PetscStrcat(binname, ".bin"));

  // Echo swarm information.
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMSwarmGetSize(swarm, &npG));
  PetscCall(DMSwarmGetLocalSize(swarm, &npL));
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] Total number of particles: %d (local) / %d (global)\n", rank, npL, npG));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  // Create the binary viewer.
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, binname, FILE_MODE_WRITE, &viewer));

  // Write particle positions to a binary file.
  PetscCall(DMSwarmCreateGlobalVectorFromField(swarm, DMSwarmPICField_coor, &target));
  PetscCall(PetscObjectSetName((PetscObject)target, "position"));
  PetscCall(VecView(target, viewer));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(swarm, DMSwarmPICField_coor, &target));

  // Destroy the binary viewer.
  PetscCall(PetscViewerDestroy(&viewer));

  // Destroy the temporary vector object.
  PetscCall(VecDestroy(&target));

  // Build the XDMF file name.
  PetscCall(PetscStrcat(xmfname, filestem));
  PetscCall(PetscStrcat(xmfname, ".xmf"));

  // Write particle positions to an XDMF file.
  PetscCall(DMSwarmViewXDMF(swarm, xmfname));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateSwarmDM(DM *swarm, DM *mesh, UserContext *user)
{
  PetscInt dim;
  PetscInt bufsize;
  PetscInt np;
  MPI_Comm comm;
  int      size;

  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "ions"));
  PetscCall(DMGetDimension(*mesh, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, *mesh));
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  PetscCall(PetscObjectGetComm((PetscObject)*mesh, &comm));
  MPI_Comm_size(comm, &size);
  if (user->particles.np < 0) {
    np = user->grid.nx * user->grid.ny * user->grid.nz;
    bufsize = np;
  } else {
    np = user->particles.np;
    bufsize = 0;
  }
  PetscCall(DMSwarmSetLocalSizes(*swarm, np / size, bufsize));

  PetscFunctionReturn(PETSC_SUCCESS);
}


#define MAXBIT 30
#define MAXDIM 6


static PetscErrorCode
Sobseq(PetscInt *n, PetscReal x[])
{
  PetscInt j, k, l;
  unsigned long i, im, ipp;
  static unsigned long in;
  static unsigned long ix[MAXDIM+1];
  static unsigned long *iu[MAXBIT+1];
  static unsigned long mdeg[MAXDIM+1]={0, 1, 2, 3, 3, 4, 4};
  static unsigned long ip[MAXDIM+1]={0, 0, 1, 1, 2, 1, 4};
  static unsigned long iv[MAXDIM*MAXBIT+1];
  static unsigned long ic[MAXDIM*MAXBIT+1]={0, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 5, 7, 7, 3, 3, 5, 15, 11, 5, 15, 13, 9};
  static PetscReal fac;

  PetscFunctionBeginUser;

  if (*n < 0) {
    for (j=1; j<=MAXDIM*MAXBIT+1; j++) {
      iv[j] = ic[j];
    }
    for (j=1, k=0; j<=MAXBIT; j++, k += MAXDIM) {
      iu[j] = &iv[k];
    }
    for (k=1; k<=MAXDIM; k++) {
      for (j=1; j<=mdeg[k]; j++) {
        iu[j][k] <<= (MAXBIT-j);
      }
      for (j=mdeg[k]+1; j<=MAXBIT; j++) {
        ipp = ip[k];
        i = iu[j-mdeg[k]][k];
        i ^= (i >> mdeg[k]);
        for (l=mdeg[k]-1; l>=1; l--) {
          if (ipp & 1) i ^= iu[j-1][k];
          ipp >>= 1;
        }
        iu[j][k] = i;
      }
    }
    fac = 1.0 / ((long)1 << MAXBIT);
    in = 0;
  } else {
    im = in++;
    for (j=1; j<=MAXBIT; j++) {
      if (!(im & 1)) break;
      im >>= 1;
    }
    if (j > MAXBIT) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "MAXBIT too small in %s", __func__);
    }
    im = (j-1)*MAXDIM;
    for (k=1; k<=PetscMin(*n, MAXDIM); k++) {
      ix[k] ^= iv[im+k];
      x[k] = ix[k]*fac;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticles(DM *swarm, UserContext *user)
{
  PetscInt    seed=-1, ndim=NDIM;
  PetscReal   *coords;
  PetscInt    np, ip;
  PetscReal   r[NDIM];
  PetscReal   L[NDIM]={user->grid.Lx, user->grid.Ly, user->grid.Lz};
  PetscInt    dim;

  PetscFunctionBeginUser;

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(*swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Initialize the psuedo-random number generator.
  PetscCall(Sobseq(&seed, r-1));

  // Get the local number of particles.
  PetscCall(DMSwarmGetLocalSize(*swarm, &np));

  for (ip=0; ip<np; ip++) {
    PetscCall(Sobseq(&ndim, r-1));
    for (dim=0; dim<NDIM; dim++) {
      coords[ip*NDIM + dim] = r[dim]*L[dim];
    }
  }

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(*swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;
  DM          mesh, swarm;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));

  // Set up discrete-mesh DM.
  PetscCall(CreateMeshDM(&mesh, &user));

  // Set up particle-swarm DM.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Creating swarm ...\n"));
  PetscCall(CreateSwarmDM(&swarm, &mesh, &user));

  // Initialize particle coordinates.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initializing particles ...\n"));
  PetscCall(InitializeParticles(&swarm, &user));

  // Move particles between ranks.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Migrating particles ...\n"));
  PetscCall(DMSwarmMigrate(swarm, PETSC_TRUE));

  // View particle coordinates.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Viewing swarm ...\n"));
  PetscCall(ViewSwarm(swarm, "coords", user));

  // Free memory.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Freeing memory ...\n"));
  PetscCall(DMDestroy(&mesh));
  PetscCall(DMDestroy(&swarm));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Finishing simulation ...\n"));
  PetscCall(PetscFinalize());

  return 0;
}

