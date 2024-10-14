#include <iostream>

#include "pumipush.h"

/**
 * Pseudo Monte Carlo simulation to simulate particle transport
 * The following steps will be performed:
 * 1. Initial source sampling
 * 2. Particles moved to new location
 * 3. Particles are tracked
 * 4. Particles are tallied
 */
int main(int argc, char* argv[]) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if (!comm_rank)
    std::cout << "PUMIPic Monte Carlo Transport Simulation on " << comm_size
              << " ranks\n";
  // ******************* Argument Parsing ******************* //

  if (argc < 6) {
    std::cerr << "Error: missing required arguments\n";
    std::cout << "Usage: " << argv[0]
              << " <mesh> <partition_file> <num_particles> <num_iterations> "
                 "<lambda>\n";
    return 1;
  }

  std::string mesh_file_name = argv[1];
  bool use_internal_box = false;
  if (mesh_file_name == "internal_box"){
	  use_internal_box = true;
  }

  // mesh file name has to be non-empty
  if (mesh_file_name.empty()) {
    std::cerr << "Error: mesh file name cannot be empty\n";
    return 1;
  }
  int num_particles = atoi(argv[3]);
  int num_iterations = atoi(argv[4]);
  double lambda = atof(argv[5]);
  std::string partition_file = argv[2];

  // ******************* Mesh Loading ******************* //
  o::Mesh full_mesh;
  if (!use_internal_box){
  	full_mesh = readMesh(mesh_file_name, lib);
  }else{
	int nx = atoi(argv[6]);
	int ny = atoi(argv[7]);
	full_mesh =  o::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 1, nx, ny, 0, false);
  }
  // o::reorder_by_hilbert(&full_mesh);
  // if (comm_rank == 0) std::cout << "Mesh Hilbert reordered\n";
  if (comm_rank == 0) {
    std::cout << "Mesh loaded sucessfully with " << full_mesh.nelems()
              << " elements\n\t\t\t"
                 "and "
              << full_mesh.nverts() << " vertices\n";
  }

  // ******** Mesh bounding box ******** //
  // Omega_h::BBox<3> bb = Omega_h::find_bounding_box<3>(full_mesh.coords());

  // ******************* Partition Loading ******************* //
  // o::HostWrite<o::LO> h_owners(full_mesh.nelems());

  o::Write<o::LO> owners(full_mesh.nelems());
  partitionMeshEqually(full_mesh, owners, comm_size, comm_rank);
#ifdef DEBUG
  printf("INFO: Owners list of the mesh created successfully\n");
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  p::Mesh picparts(full_mesh, owners);
  printf("Partitioned the mesh successfully\n");

  o::Mesh* mesh = picparts.mesh();

  /// Omega_h::vtk::write_parallel("initialPartition.vtk", mesh, 2);
  // std::cout << "Initial partition written to initialPartition.vtk\n";

  // ******************* Particle Initialization ******************* //
  Omega_h::Int ne = mesh->nelems();

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  // printf("Number of particles: \t%d in rank \t%d\n", num_particles,
  // comm_rank);
  printf("INFO: Number of elements: \t%d in rank %d\n", ne, comm_rank);
#endif

  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(  // ? what is this doing? copying from GOs to kkGidView
      ne,
      OMEGA_H_LAMBDA(const int& i) { element_gids(i) = mesh_element_gids[i]; });

  // int numPtcls = setSourceElements(picparts, ptcls_per_elem, 0,
  // num_particles);
  int total_num_particles = num_particles;
  // if (comm_rank != 0) num_particles = 0;
  {
    double element_fraction = double(ne) / full_mesh.nelems();
    num_particles = int(num_particles * element_fraction);
  }
  // int setPtcls =
  //     distributeParticlesEqually(picparts, ptcls_per_elem, num_particles);
  int setPtcls =
      distributeParticlesBasesOnArea(picparts, ptcls_per_elem, num_particles);
  printf("INFO: Number of particles set to elements: \t %d in rank %d\n",
         setPtcls, comm_rank);
  Kokkos::TeamPolicy<ExeSpace> policy;
#ifdef PP_USE_GPU
  printf("Using GPU for simulation...");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#else
  printf("Using CPU for simulation...");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, 1);
#endif

  // Sell-C-Sigma particle structure: see the pumipic paper for more details
  //PS* ptcls = new SellCSigma<Particle>(policy, INT_MAX, 1024, ne, setPtcls,
  //                                     ptcls_per_elem, element_gids);
  PS* ptcls = new p::DPS<Particle>(policy, ne, setPtcls, ptcls_per_elem, element_gids);
#ifdef DEBUG
  printf("INFO: Particle structure created successfully in rank %d\n",
         comm_rank);
#endif
  random_pool_t random_pool(347932874);
  // setInitialPtclCoords(picparts, ptcls, random_pool);
  setUniformPtclCoords(picparts, ptcls, random_pool);
#ifdef DEBUG
  printf("INFO: Initial positions of particles set... in rank %d\n", comm_rank);
#endif
  setPtclIds(ptcls);
#ifdef DEBUG
  printf("INFO: Particles initialized with ids... in rank %d\n", comm_rank);
#endif

  o::LOs elmTags(ne, -1, "elmTagVals");
  o::Write<o::Real> flux(mesh->nelems(), 0.0);
  mesh->add_tag(o::FACE, "has_particles", 1,
                elmTags);  // ncomp is number of components
  mesh->add_tag(o::VERT, "avg_density", 1, o::Reals(mesh->nverts(), 0));
  //render(picparts, 0, comm_rank);

  {
    computeAvgPtclDensity(picparts, ptcls, flux);
    render(picparts, 0, comm_rank);
  }

  Kokkos::Timer timer;
  Kokkos::Timer fullTimer;
  Kokkos::Timer iterTimer;

  int iter=0;  // iteration number
  int np;    //
  int ps_np=0;
  random_pool_t rand_pool(1);
  o::Write<o::LO> elem_ids(ptcls->capacity(), -1, "elem_ids");
  

  // ******************* Monte Carlo Transport Simulation ******************* //
  //for (iter = 1; iter <= num_iterations; ++iter) {
  do {
    ++iter;
    iterTimer.reset();
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank == 0) printf("-----------------------------\n");
    printf("Iteration: %d Rank: %d\n", iter, comm_rank);
    // 1. check the remaining number of particles
    ps_np = ptcls->nPtcls();
    std::cout << "Number of particles: " << ps_np << " in rank " << comm_rank
              << '\n';
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    timer.reset();
    // 2. push particles
    pseudo2Dpush(ptcls, lambda, rand_pool);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank == 0)
      fprintf(stderr, "TIME: push and transfer (seconds) %f\n",
              timer.seconds());
    timer.reset();
    // 3. search for the new element
    search(picparts, ptcls, flux, elem_ids, false);
    if (comm_rank == 0)
      fprintf(stderr, "search, rebuild, and transfer (seconds) %f\n",
              timer.seconds());
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


    double iter_time = iterTimer.seconds();
    printf("TIME: Iteration %d took %f seconds for %d particles(time for 1000 particles=%f)\n", iter, iter_time, ps_np, (1000.0 * iter_time)/ps_np);
    if (np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if (iter >= num_iterations){
	    printf("Iter %d is done and it's the max iter (%d)", iter, num_iterations);
	    break;
    }
    // 4. tag the parent elements
    // tagParentElements(picparts, ptcls, iter);
    //computeAvgPtclDensity(picparts, ptcls, flux);
    //render(picparts, iter, comm_rank);
  } while(ps_np > 0.01*setPtcls);
  computeFluxAndAdd(picparts, flux, iter - 1);
  render(picparts, iter, comm_rank);

  if (comm_rank == 0) {
    fprintf(stderr, "%d iterations of pseudopush (seconds) %f\n", iter - 1,
            fullTimer.seconds());
  }

  // cleanup
  delete ptcls;

  pumipic::SummarizeTime();
  if (!comm_rank) fprintf(stderr, "done\n");
  return 0;
}
