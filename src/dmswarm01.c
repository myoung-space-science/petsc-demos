static char help[] = "Examine DMSWARM particle initialzation methods.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


#define NDIM 3

typedef struct {
  PetscInt nx, ny, nz;
  PetscReal Lx, Ly, Lz;
  PetscReal x0, y0, z0;
  PetscReal x1, y1, z1;
  DMBoundaryType bc;
} Grid;

typedef struct {
  PetscInt np;
  PetscInt npc;
  PetscInt npd[NDIM];
} Particles;

typedef struct {
  Grid grid;
  Particles particles;
  PetscReal dx, dy, dz;
  PetscBool remove;
} UserContext;


static PetscErrorCode
ProcessOptions(UserContext *options)
{
  PetscFunctionBeginUser;

  PetscInt  intArg;
  PetscReal realArg;
  PetscBool boolArg, found;
  char      strArg[PETSC_MAX_PATH_LEN];

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
  options->dx = 0.0;
  options->dy = 0.0;
  options->dz = 0.0;
  options->remove = PETSC_FALSE;

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
    options->particles.np = options->grid.nx * options->grid.ny * options->grid.nz;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npc", &intArg, &found));
  if (found) {
    options->particles.npc = intArg;
  } else {
    options->particles.npc = 1;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npd", &intArg, &found));
  if (found) {
    options->particles.npd[0] = intArg;
    options->particles.npd[1] = intArg;
    options->particles.npd[2] = intArg;
  } else {
    options->particles.npd[0] = options->grid.nx;
    options->particles.npd[1] = options->grid.ny;
    options->particles.npd[2] = options->grid.nz;
  }
  PetscCall(PetscOptionsGetString(NULL, NULL, "-bc", strArg, sizeof(strArg), &found));
  if (found && (strcmp(strArg, "periodic")==0)) {
    options->grid.bc = DM_BOUNDARY_PERIODIC;
  } else {
    options->grid.bc = DM_BOUNDARY_GHOSTED;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dx", &realArg, &found));
  if (found) {
    options->dx = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dy", &realArg, &found));
  if (found) {
    options->dy = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dz", &realArg, &found));
  if (found) {
    options->dz = realArg;
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "--remove", &boolArg, &found));
  if (found) {
    options->remove = boolArg;
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
CreateSwarmDM(DM *swarm, DM *mesh, UserContext *user)
{
  PetscInt dim;
  PetscInt bufsize=0;
  PetscInt np;
  MPI_Comm comm;
  int      size;

  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "Ions"));
  PetscCall(DMGetDimension(*mesh, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, *mesh));
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*swarm, "potential", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*swarm, "density", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  PetscCall(PetscObjectGetComm((PetscObject)*mesh, &comm));
  MPI_Comm_size(comm, &size);
  np = user->particles.np / size;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Setting local sizes <<<\n"));
  PetscCall(DMSwarmSetLocalSizes(*swarm, np, bufsize));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ShiftParticles(DM swarm, UserContext user, PetscBool rmpart)
{
  PetscReal *coords;
  PetscInt   ip, np;
  PetscReal  dx=user.dx;
  PetscReal  dy=user.dy;
  PetscReal  dz=user.dz;
  PetscReal  Lx=user.grid.Lx;
  PetscReal  Ly=user.grid.Ly;
  PetscReal  Lz=user.grid.Lz;
  PetscReal  x, y, z;

  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetLocalSize(swarm, &np));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Shifting particle positions by (%g, %g, %g) <<<\n", dx, dy, dz));
  for (ip=0; ip<np; ip++) {
    x = coords[ip*NDIM + 0];
    y = coords[ip*NDIM + 1];
    z = coords[ip*NDIM + 2];
    x += dx;
    while (x > Lx) {x -= Lx;}
    while (x < 0)  {x += Lx;}
    y += dy;
    while (y > Ly) {y -= Ly;}
    while (y < 0)  {y += Ly;}
    z += dz;
    while (z > Lz) {z -= Lz;}
    while (z < 0)  {z += Lz;}
    coords[ip*NDIM + 0] = x;
    coords[ip*NDIM + 1] = y;
    coords[ip*NDIM + 2] = z;
  }
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(swarm, rmpart));
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating again <<<\n"));
  PetscCall(DMSwarmMigrate(swarm, rmpart));
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticlesFromCellDM(DM *swarm, UserContext *user, PetscBool rmpart)
{
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Inserting points using cell DM <<<\n"));
  PetscCall(DMSwarmInsertPointsUsingCellDM(*swarm, DMSWARMPIC_LAYOUT_REGULAR, user->particles.npc));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, rmpart));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ShiftParticles(*swarm, *user, rmpart));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticlesFromCoordinates(DM *swarm, UserContext *user, PetscBool rmpart)
{
  PetscReal min[NDIM], max[NDIM];
  PetscInt ndir[NDIM];

  PetscFunctionBeginUser;

  min[0] = user->grid.x0;
  max[0] = user->grid.x1;
  min[1] = user->grid.y0;
  max[1] = user->grid.y1;
  min[2] = user->grid.z0;
  max[2] = user->grid.z1;
  ndir[0] = user->particles.npd[0];
  ndir[1] = user->particles.npd[1];
  ndir[2] = user->particles.npd[2];

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Inserting points using uniform coordinates <<<\n"));
  PetscCall(DMSwarmSetPointsUniformCoordinates(*swarm, min, max, ndir, INSERT_VALUES));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, rmpart));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ShiftParticles(*swarm, *user, rmpart));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;
  DM          mesh, swarm;
  char        rmstr[2048]="";

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n************************* START **********************\n\n"));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));

  // Set up discrete mesh.
  PetscCall(CreateMeshDM(&mesh, &user));

  // Set up particle swarm.
  PetscCall(CreateSwarmDM(&swarm, &mesh, &user));

  // Set initial particle positions and velocities.
  if (user.remove) {
    strcpy(rmstr, "-------------------- WITH REMOVAL -----------------------");
  } else {
    strcpy(rmstr, "-------------------- WITHOUT REMOVAL --------------------");
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n%s\n\n", rmstr));
  PetscCall(InitializeParticlesFromCellDM(&swarm, &user, user.remove));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n%s\n\n", rmstr));
  PetscCall(InitializeParticlesFromCoordinates(&swarm, &user, user.remove));

  // Free memory.
  PetscCall(DMDestroy(&mesh));
  PetscCall(DMDestroy(&swarm));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n************************* END *************************\n"));
  PetscCall(PetscFinalize());

  return 0;
}