/**
 * Pseudo Monte Carlo simulation to simulate particle transport
 * The following steps will be performed:
 * 1. Initial source sampling
 * 2. Particles moved to new location
 * 3. Particles are tracked
 * 4. Particles are tallied
 */

#include <Omega_h_macros.h>
#include <ppMacros.h>

#include <Kokkos_Core.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mark.hpp>
#include <Omega_h_mesh.hpp>
#include <memory>
#include <pumipic_adjacency.hpp>
#include <pumipic_adjacency.tpp>
#include <variant>
// #include <pumipic_kktypes.hpp>
#include <pumipic_mesh.hpp>
#include <pumipic_ptcl_ops.hpp>
#include <random>

#define PARTICLE_SEED 10

namespace o = Omega_h;
namespace p = pumipic;

using p::fp_t;
using p::Vector3d;
/* Define particle types
   0 = current position
   1 = pushed position
   2 = ids
   3 = velocity
 */
typedef p::MemberTypes<Vector3d, Vector3d, int, Vector3d> Particle;
typedef p::ParticleStructure<Particle> PS;
typedef Kokkos::DefaultExecutionSpace ExeSpace;

typedef Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>
    random_pool_t;
using random_state_t = Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace>;

int setSourceElements(o::Mesh mesh, PS::kkLidView ppe, const int numPtcls) {
  auto numPpe = numPtcls / mesh.nelems();
  auto numPpeR = numPtcls % mesh.nelems();
  auto cells2nodes = mesh.ask_down(mesh.dim(), o::VERT).ab2b;
  auto nodes2coords = mesh.coords();

  o::parallel_for(
      mesh.nelems(),
      OMEGA_H_LAMBDA(const int i) { ppe[i] = numPpe + (i < numPpeR); });
  Omega_h::LO totPtcls = 0;
  Kokkos::parallel_reduce(
      ppe.size(),
      OMEGA_H_LAMBDA(const int i, Omega_h::LO &lsum) { lsum += ppe[i]; },
      totPtcls);

  assert(totPtcls == numPtcls);
  return totPtcls;
}

PS *create_particle_structure(o::Mesh mesh, p::lid_t &numPtcls) {
  Omega_h::Int ne = mesh.nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::parallel_for(
      ne, OMEGA_H_LAMBDA(const int &i) { element_gids(i) = i; });
  numPtcls = setSourceElements(mesh, ptcls_per_elem, numPtcls);
  Omega_h::parallel_for(
      ne, OMEGA_H_LAMBDA(const int &i) { const int np = ptcls_per_elem(i); });

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

#ifdef PP_USE_GPU
  printf("[INFO] Using GPU for simulation...");
  policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#else
  printf("Using CPU for simulation...");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, 32);
#endif

  // needs pumipic with cabana
  PS *ptcls =
      new p::DPS<Particle>(policy, ne, numPtcls, ptcls_per_elem, element_gids);
  printf("[INFO] DPS Particle structure created successfully\n");
  //'sigma', 'V', and the 'policy' control the layout of the PS structure
  // in memory and can be ignored until performance is being evaluated.  These
  // are reasonable initial settings for OpenMP.
  // const int sigma = INT_MAX; // full sorting
  // const int V = 1024;
  // PS *ptcls = PS *ptcls = new p::SellCSigma<Particle>(
  //    policy, sigma, V, ne, actualParticles, ptcls_per_elem, element_gids);
  // printf("SellCSigma Particle structure created successfully\n");
  return ptcls;
}

/*
 * Sets initial particle positions, ids, and velocities(not used)
 */
void initializeUniformPtclCoordsAndVelocity(p::Mesh &picparts, PS *ptcls,
                                            random_pool_t random_pool) {
  o::Mesh *mesh = picparts.mesh();
  int dim = mesh->dim();
  OMEGA_H_CHECK(dim == 3);

  auto cells2nodes = mesh->ask_down(o::REGION, o::VERT).ab2b;
  auto coords = mesh->coords();

  auto x_ps_d = ptcls->get<0>();
  auto x_ps_tgt_d = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  auto v_ps_d = ptcls->get<3>();

  auto set_initial_positions =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0) {
      pids(pid) = pid;

      auto gen = random_pool.get_state();
      o::Real x = gen.drand(0.0, 1.0);
      o::Real y = gen.drand(0.0, 1.0);
      o::Real z = gen.drand(0.0, 1.0);
      o::Real angle = gen.drand(0.0, 1.0) * 2.0 * M_PI;
      o::Real r = gen.drand(0.0, 1.0) * 2.0 - 1.0;
      random_pool.free_state(gen);

      if (x + y > 1) {
        x = 1 - x;
        y = 1 - y;
      }
      if (y + z > 1) {
        o::Real tmp = z;
        z = 1 - x - y;
        y = 1 - tmp;
      } else if (x + y + z > 1) {
        o::Real tmp = z;
        z = x + y + z - 1;
        x = 1 - y - tmp;
      }

      const auto verts = o::gather_verts<4>(cells2nodes, e);
      const auto vtxCoords = o::gather_vectors<4, 3>(coords, verts);

      const o::Real a = 1 - x - y - z;
      for (int i = 0; i < 3; i++) {
        x_ps_d(pid, i) = a * vtxCoords[0][i] + x * vtxCoords[1][i] +
                         y * vtxCoords[2][i] + z * vtxCoords[3][i];
        // x_ps_tgt_d(pid, i) = x_ps_d(pid, i);
      }
      v_ps_d(pid, 0) = Kokkos::sqrt(1 - r * r) * Kokkos::cos(angle);
      v_ps_d(pid, 1) = Kokkos::sqrt(1 - r * r) * Kokkos::sin(angle);
      v_ps_d(pid, 2) = r;

      // TODO - debug remove later
      // if (pid == 0) {
      //  printf("Initial position: %f %f %f\n", x_ps_d(pid, 0), x_ps_d(pid, 1),
      //         x_ps_d(pid, 2));
      //}
    }
  };
  ps::parallel_for(ptcls, set_initial_positions);
}

// Push particles
void push_ptcls(PS *ptcls, o::Real lambda, random_pool_t &random_pool) {
  Kokkos::Timer timer;
  timer.reset();
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto angle = ptcls->get<3>();

  auto push =
      PS_LAMBDA(const p::lid_t elm, const p::lid_t ptcl, const bool mask) {
    if (mask) {
      auto gen = random_pool.get_state();
      o::Real phi = gen.drand(0.0, 1.0) * 2 * M_PI;
      o::Real z = gen.drand(-1.0, 1.0);
      o::Real theta = Kokkos::acos(z);
      o::Real rn = gen.drand(0.0, 1.0);
      random_pool.free_state(gen);

      angle(ptcl, 0) = Kokkos::sin(theta) * Kokkos::cos(phi);
      angle(ptcl, 1) = Kokkos::sin(theta) * Kokkos::sin(phi);
      angle(ptcl, 2) = Kokkos::cos(theta);

      o::Real distance = -Kokkos::log(rn) * lambda;
      for (int i = 0; i < 3; ++i) {
        tgt(ptcl, i) = cur(ptcl, i) + (distance * angle(ptcl, i));
      }

      // TODO - debug remove later
      // if (ptcl == 0) {
      //   printf("** Distance: %f\n", distance);
      //   printf("** Current position: %f %f %f\n", cur(ptcl, 0), cur(ptcl, 1),
      //   cur(ptcl, 2)); printf("** Pushed position:  %f %f %f\n", tgt(ptcl,
      //   0), tgt(ptcl, 1),  tgt(ptcl, 2));
      // }
    }
  };
  p::parallel_for(ptcls, push, "push");

  double totTime = timer.seconds();
  printf("[TIME] push: %f\n", totTime);
}

void updatePtclPositions(PS *ptcls) {
  Kokkos::Timer timer;
  timer.reset();
  double totTime = 0.;
  auto x_ps_d = ptcls->get<0>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto updatePtclPos = PS_LAMBDA(const int &, const int &pid, const bool &) {
    x_ps_d(pid, 0) = xtgt_ps_d(pid, 0);
    x_ps_d(pid, 1) = xtgt_ps_d(pid, 1);
    x_ps_d(pid, 2) = xtgt_ps_d(pid, 2);
    xtgt_ps_d(pid, 0) = 0.0;
    xtgt_ps_d(pid, 1) = 0.0;
    xtgt_ps_d(pid, 2) = 0.0;
  };
  ps::parallel_for(ptcls, updatePtclPos);

  totTime += timer.seconds();
  printf("[TIME] updatePtclPosition: %f\n", totTime);
}

void rebuild(p::Mesh &picparts, PS *ptcls, o::LOs elem_ids) {
  Kokkos::Timer timer;
  timer.reset();
  double totTime = 0.0;

  updatePtclPositions(ptcls);
  p::migrate_lb_ptcls(picparts, ptcls, elem_ids, 1.05);
  p::printPtclImb(ptcls);

  totTime = timer.seconds();
  printf("[TIME] rebuild: %f\n", totTime);
}

// read mesh (gmsh or osh)
o::Mesh readMesh(const char *meshFileName, o::Library &lib) {
  // check the extension of the mesh file
  std::string meshFile(meshFileName);
  std::string ext = meshFile.substr(meshFile.find_last_of(".") + 1);
  if (ext == "osh") {
    return o::binary::read(meshFileName, lib.self());
  } else if (ext == "msh") {
    return o::gmsh::read(meshFileName, lib.self());
  } else {
    std::cerr << "Error: unsupported mesh file format. Found format **%s**.\n",
        ext.c_str();
    exit(1);
  }
}

// void apply_reflective_boundary_condition(o::Mesh &mesh, PS *ptcls,
//                                          o::Write<o::LO> &elem_ids,
//                                          o::Write<o::LO> &ptcl_done,
//                                          o::Write<o::LO> &lastExit,
//                                          o::Write<o::LO> &xFace) {
//   const auto &side_is_exposed = o::mark_exposed_sides(&mesh);
//
//   auto apply_reflective_bc = PS_LAMBDA(const int e, const int pid, const int
//   mask) {
//     if (mask > 0 && !ptcl_done[pid]) {
//       assert(lastExit[pid] != -1);
//
//     }
//
//
//
// }

void apply_vacuum_boundary_condition(o::Mesh &mesh, PS *ptcls,
                                     o::Write<o::LO> &elem_ids,
                                     o::Write<o::LO> &ptcl_done,
                                     o::Write<o::LO> &lastExit,
                                     o::Write<o::LO> &xFace) {
  const auto &side_is_exposed = o::mark_exposed_sides(&mesh);

  auto checkExposedEdges =
      PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      assert(lastExit[pid] != -1);
      const o::LO bridge = lastExit[pid];
      const bool exposed = side_is_exposed[bridge];
      ptcl_done[pid] = exposed;
      xFace[pid] = lastExit[pid];
      elem_ids[pid] = exposed ? -1 : elem_ids[pid];
    }
  };
  p::parallel_for(ptcls, checkExposedEdges, "apply vacumm boundary condition");
}

void move_to_new_element(o::Mesh &mesh, PS *ptcls, o::Write<o::LO> &elem_ids,
                         o::Write<o::LO> &ptcl_done,
                         o::Write<o::LO> &lastExit) {
  const int dim = mesh.dim();
  const auto &face2elems = mesh.ask_up(dim - 1, dim);
  const auto &face2elemElem = face2elems.ab2b;
  const auto &face2elemOffset = face2elems.a2ab;

  auto set_next_element =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      auto searchElm = elem_ids[pid];
      auto bridge = lastExit[pid];
      auto e2f_first = face2elemOffset[bridge];
      auto e2f_last = face2elemOffset[bridge + 1];
      auto upFaces = e2f_last - e2f_first;
      assert(upFaces == 2);
      auto faceA = face2elemElem[e2f_first];
      auto faceB = face2elemElem[e2f_first + 1];
      assert(faceA != faceB);
      assert(faceA == searchElm || faceB == searchElm);
      auto nextElm = (faceA == searchElm) ? faceB : faceA;
      elem_ids[pid] = nextElm;
    }
  };
  parallel_for(ptcls, set_next_element, "pumipic_set_next_element");
}

struct ParticleAtBoundary {
  ParticleAtBoundary(o::LO nelems, o::LO capacity)
      : flux_(nelems, 0.0, "flux"),
        prev_xpoint_(capacity * 3, 0.0, "prev_xpoint") {
    printf(
        "[INFO] Particle handler at boundary with %d elements and %d "
        "particles\n",
        flux_.size(), prev_xpoint_.size());
  }

  void operator()(o::Mesh &mesh, PS *ptcls, o::Write<o::LO> &elem_ids,
                  o::Write<o::LO> &inter_faces, o::Write<o::LO> &lastExit,
                  o::Write<o::Real> &inter_points, o::Write<o::LO> &ptcl_done) {
    apply_vacuum_boundary_condition(mesh, ptcls, elem_ids, ptcl_done, lastExit,
                                    inter_faces);
    move_to_new_element(mesh, ptcls, elem_ids, ptcl_done, lastExit);
    evaluateFlux(ptcls, inter_points);
  }

  void updatePrevXPoint(o::Write<o::Real> &xpoints) {
    OMEGA_H_CHECK_PRINTF(
        xpoints.size() <= prev_xpoint_.size() && prev_xpoint_.size() != 0,
        "xpoints size %d is greater than prev_xpoint size %d\n", xpoints.size(),
        prev_xpoint_.size());
    auto prev_xpoint = prev_xpoint_;
    auto update = OMEGA_H_LAMBDA(o::LO i) { prev_xpoint[i] = xpoints[i]; };
    o::parallel_for(xpoints.size(), update, "update previous xpoints");
  }

  void evaluateFlux(PS *ptcls, o::Write<o::Real> &xpoints);
  o::Reals normalizeFlux(o::Mesh &mesh);

  o::Write<o::Real> flux_;
  o::Write<o::Real> prev_xpoint_;
};

void ParticleAtBoundary::evaluateFlux(PS *ptcls, o::Write<o::Real> &xpoints) {
  o::Real total_particles = ptcls->nPtcls();
  auto prev_xpoint = prev_xpoint_;
  auto flux = flux_;

  auto evaluate_flux =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0) {
      o::Vector<3> dest = {xpoints[pid * 3], xpoints[pid * 3 + 1],
                           xpoints[pid * 3 + 2]};
      o::Vector<3> orig = {prev_xpoint[pid * 3], prev_xpoint[pid * 3 + 1],
                           prev_xpoint[pid * 3 + 2]};

      o::Real parsed_dist = o::norm(dest - orig);  // / total_particles;
      Kokkos::atomic_add(&flux[e], parsed_dist);
    }
  };
  p::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
}

o::Reals ParticleAtBoundary::normalizeFlux(o::Mesh &mesh) {
  const o::LO nelems = mesh.nelems();
  const auto &el2n = mesh.ask_down(o::REGION, o::VERT).ab2b;
  const auto &coords = mesh.coords();

  auto flux = flux_;

  o::Write<o::Real> tet_volumes(flux_.size(), -1.0, "tet_volumes");
  o::Write<o::Real> normalized_flux(flux_.size(), -1.0, "normalized flux");

  auto normalize_flux_with_volume = OMEGA_H_LAMBDA(o::LO elem_id) {
    const auto elem_verts = o::gather_verts<4>(el2n, elem_id);
    const auto elem_vert_coords = o::gather_vectors<4, 3>(coords, elem_verts);

    auto b = o::simplex_basis<3, 3>(elem_vert_coords);
    auto volume = o::simplex_size_from_basis(b);

    tet_volumes[elem_id] = volume;
    normalized_flux[elem_id] = flux[elem_id] / volume;
  };
  o::parallel_for(tet_volumes.size(), normalize_flux_with_volume,
                  "normalize flux");

  mesh.add_tag(o::REGION, "volume", 1, o::Reals(tet_volumes));
  return o::Reals(normalized_flux);
}

void print_exposed_faces(p::Mesh &picparts) {
  o::Mesh *mesh = picparts.mesh();
  const auto side_is_exposed = mark_exposed_sides(mesh);
  const auto &face2node = mesh->ask_down(o::FACE, o::VERT).ab2b;

  auto print_exposed = OMEGA_H_LAMBDA(o::LO face) {
    if (side_is_exposed[face]) {
      auto nodes = o::gather_verts<3>(face2node, face);
      printf("Exposed face %d: %d %d %d\n", face, nodes[0], nodes[1], nodes[2]);
    }
  };
  o::parallel_for(mesh->nfaces(), print_exposed);

  // get total number of exposed faces
  o::LO totalExposedFaces = o::get_sum(side_is_exposed);
  printf("Total exposed faces: %d out of %d\n", totalExposedFaces,
         mesh->nfaces());

  printf("Exposed faces printed. Exiting...\n");
  exit(1);
}

bool search(p::Mesh &picparts, PS *ptcls, o::Write<o::LO> &elem_ids,
            o::Write<o::Real> &inter_points, o::Write<o::LO> &inter_faces,
            ParticleAtBoundary &handle_particle_at_elem_boundary) {
  o::Mesh *mesh = picparts.mesh();
  OMEGA_H_CHECK(ptcls->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 10000;
  const auto psCapacity = ptcls->capacity();

  auto x = ptcls->get<0>();
  auto xtgt = ptcls->get<1>();
  auto pid = ptcls->get<2>();

  bool isFound = p::particle_search(*mesh, ptcls, x, xtgt, pid, elem_ids,
                                    inter_faces, inter_points, maxLoops,
                                    handle_particle_at_elem_boundary);
  handle_particle_at_elem_boundary.updatePrevXPoint(inter_points);

  rebuild(picparts, ptcls, elem_ids);
  return isFound;
}

int main(int argc, char **argv) {
  // ******************* Initialization ******************* //
  Kokkos::Timer timer;
  timer.reset();

  p::Library pic_lib(&argc, &argv);
  o::Library &lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if (!comm_rank) {
    printf("PUMIPic Monte Carlo Transport Simulation on %d ranks\n", comm_size);
  }

  if (argc != 4) {
    printf("Usage: %s <mesh> <num_particles> <lambda>\n", argv[0]);
    return 1;
  }
  const char *meshFileName = argv[1];
  int numPtcls = atoi(argv[2]);
  const o::Real lambda = atof(argv[3]);
  printf("Initializing simulation with mesh %s, %d particles, and lambda %f\n",
         meshFileName, numPtcls, lambda);

  o::Mesh full_mesh = readMesh(meshFileName, lib);
  // TODO - create a partitioned mesh
  o::LOs owners(full_mesh.nelems(), 0, "owners");
  p::Mesh picparts(full_mesh, owners);
  // print_exposed_faces(picparts);
  printf("Partitioned the mesh successfully\n");
  o::Mesh *mesh = picparts.mesh();
  o::LO ne = mesh->nelems();
  printf("Mesh %s loaded with %d elements\n", meshFileName, ne);

  PS *ptcls = create_particle_structure(*mesh, numPtcls);
  o::LO capacity = ptcls->capacity();
  printf("Particle Structure created with %d particles and %d capacity\n",
         numPtcls, capacity);
  random_pool_t random_pool(PARTICLE_SEED);
  initializeUniformPtclCoordsAndVelocity(picparts, ptcls, random_pool);
  printf("Particle initialization complete\n");

  o::Real time = timer.seconds();
  printf("[TIME] Time to initialize simulation: %f\n", time);

  // ******************* Monte Carlo Transport Simulation ******************* //
  o::LO maxIter = 100000;
  o::LO np;
  o::LO iter = 0;
  o::Real timePerIter = 0.0;
  o::Write<o::LO> elem_ids;
  // elem_ids size has to be zero to start with for search_mesh requirement
  OMEGA_H_CHECK_PRINTF(elem_ids.size() == 0,
                       "[ERROR] elem_ids size has to be zero but found %d\n",
                       elem_ids.size());
  o::Write<o::Real> inter_points;
  o::Write<o::LO> inter_faces;

  ParticleAtBoundary particleAtBoundaryManager(ne, capacity);

  do {
    Kokkos::fence();
    np = ptcls->nPtcls();
    printf(
        "\n------------------------------Iteration %d ----------------------\n",
        iter);
    printf("[INFO] Iteration %d started on Rank: %d with %d particles\n", iter,
           comm_rank, np);

    push_ptcls(ptcls, lambda, random_pool);
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);
    bool found = search(picparts, ptcls, elem_ids, inter_points, inter_faces,
                        particleAtBoundaryManager);
    if (!found) {
      printf("[ERROR] Particle search failed\n");
      exit(1);
    }

    // TODO - debug section, delete it later
    // auto elem_ids_h = o::HostRead<o::LO>(elem_ids);
    // printf("** After %d iteration elem_ids[0] = %d\n", iter, elem_ids_h[0]);

    np = ptcls->nPtcls();
    timePerIter = timer.seconds();
    printf(
        "[INFO] Iteration %d completed on Rank: %d with %d particles in %f "
        "seconds.\n\n\n\n",
        iter, comm_rank, np, timePerIter);

    iter++;
  } while (np > 0.0001 * numPtcls && iter < maxIter);

  time = timer.seconds();
  printf(
      "[INFO] Simulation completed after %d iterations in %f seconds in "
      "total.\n",
      iter, time);

  const char *result_vtk_fname = "results.vtk";
  printf("[STATUS] Normalizing the flux with volume and writing it to %s\n",
         result_vtk_fname);
  o::Reals flux = particleAtBoundaryManager.normalizeFlux(*mesh);
  mesh->add_tag(o::REGION, "flux", 1, flux);

  o::vtk::write_parallel(result_vtk_fname, mesh, 3);

  delete ptcls;
  return 0;
}