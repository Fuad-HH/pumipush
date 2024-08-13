#ifndef PUMIPUSH_H
#define PUMIPUSH_H

#include <Omega_h_macros.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Random.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>
#include <cmath>
#include <fstream>
#include <particle_structs.hpp>
#include <pumipic_kktypes.hpp>
#include <random>
// #include <redev_partition.h>

// #include "redev.h"

#include <Segment.h>
#include <ppMacros.h>

#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_ptcl_ops.hpp"

// #include <Kokkos_Core_fwd.hpp>
#include <Kokkos_MinMax.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_vector.hpp>
#include <cstdlib>
#include <optional>
#include <particle_structure.hpp>
#include <pumipic_adjacency.tpp>
#include <pumipic_constants.hpp>

#define NUM_ITERATIONS 30

using particle_structs::lid_t;
using particle_structs::MemberTypes;
using particle_structs::SellCSigma;
using pumipic::fp_t;
using pumipic::Vector3d;

namespace o = Omega_h;
namespace p = pumipic;
namespace ps = particle_structs;

// To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
//  computed (pre adjacency search) positions, and
//-an integer to store the particles id
typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef ps::ParticleStructure<Particle> PS;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>
    random_pool_t;

struct IntersectionResult {
  bool exists = false;
  o::Vector<2> point = o::Vector<2>{0.0, 0.0};
};
// ******************** Function Prototypes ******************** //
Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> TeamPolicyAutoSelect(
    int league_size, int team_size);

OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors(o::Reals const& a,
                                             o::Few<o::LO, 4> v);

/**
 * @brief Generate a random path length using an exponential distribution
 * TODO: make this lambda a mean free path length
 */
OMEGA_H_DEVICE double random_path_length(double lambda, random_pool_t pool);

o::Mesh readMesh(std::string meshFile, o::Library& lib);

/**
 * Populate the particles equally to all elements
 */
int distributeParticlesEqually(const p::Mesh& picparts, PS::kkLidView ppe,
                               const int numPtcls);

/**
 * Set the initial particle coordinates to the centroid of the parent element
 */
void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls,
                          random_pool_t random_pool);

/**
 * Set the particle ids to the particle index
 */
void setPtclIds(PS* ptcls);

/**
 * Render the mesh with the particles
 */
void render(p::Mesh& picparts, int iter, int comm_rank);

/**
 * Print the timing of the operation
 */
void printTiming(const char* name, double t);

/**
 * Push the particles in the direction of the vector
 */
void pseudo2Dpush(PS* ptcls, double lambda);

void push(PS* ptcls, int np, double lambda);

/**
 * Get a random direction uniformly distributed on the unit sphere
 */
OMEGA_H_DEVICE o::Vector<3> sampleRandomDirection(const double A,
                                                  random_pool_t random_pool);

/**
 * Update the particle positions to the new target positions
 */
void updatePtclPositions(PS* ptcls);

/**
 * Rebuild the particle structure with the new element ids
 * It just calls the *migrate_lb_ptcls* function
 */
void rebuild(p::Mesh& picparts, PS* ptcls, o::LOs elem_ids, const bool output);

/**
 * Search for the new element for the particles
 */
void search(p::Mesh& picparts, PS* ptcls, bool output);

/**
 * Tag the parent elements with the number of particles
 */
void tagParentElements(p::Mesh& picparts, PS* ptcls, int loop);

/**
 * Create owner list based on the cpn file
 */
void ownerFromCPN(const std::string cpn_file_name, o::Write<o::LO>& owners);

/**
 * @brief partition mesh equally based on the number of ranks
 */
void partitionMeshEqually(o::Mesh& mesh, o::Write<o::LO> owners, int comm_size,
                          int comm_rank);

/**
 * \brief this function is used to get partitions of mesh only in the xy plane
 */
void get_bounding_box_in_xy_plane(Omega_h::Mesh& mesh, o::Vector<2>& min,
                                  o::Vector<2>& max);

void create_int_rectangle(const int total, int& nrows, int& ncols);

void varify_balance_of_partitions(o::Write<o::LO>& owners, int comm_size);

template <typename T>
void prettyPrintBB(T min, T max);

OMEGA_H_DEVICE void get_tri_centroid(const o::LOs& cells2nodes, o::LO e,
                                     const o::Reals& nodes2coords,
                                     o::Few<o::Real, 2>& center);

OMEGA_H_DEVICE void get_tet_centroid(const o::LOs& cells2nodes, o::LO e,
                                     const o::Reals& nodes2coords,
                                     o::Few<o::Real, 3>& center);

OMEGA_H_DEVICE void cylindrical2cartesian(const o::Vector<3> cyl,
                                          o::Vector<3>& cartesian);

OMEGA_H_DEVICE void cartesian2cylindrical(const o::Vector<3> cartesian,
                                          o::Vector<3>& cyl);
template <class Vec>
OMEGA_H_DEVICE bool all_positive(const Vec a, Omega_h::Real tol = EPSILON) {
  auto isPos = 1;
  for (Omega_h::LO i = 0; i < a.size(); ++i) {
    const auto gtez = Omega_h::are_close(a[i], 0.0, tol, tol) || a[i] > 0;
    isPos = isPos && gtez;
  }
  return isPos;
}

OMEGA_H_DEVICE IntersectionResult find_intersection_point(
    o::Few<o::Vector<2>, 2> line1, o::Few<o::Vector<2>, 2> line2);

void computeAvgPtclDensity(p::Mesh& picparts, PS* ptcls);

OMEGA_H_DEVICE double distance_between_points(o::Vector<2> p1,
                                              o::Vector<2> p2) {
  return o::norm(p1 - p2);
}

/// ref: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
OMEGA_H_DEVICE double find_intersection_distance_tri(
    const Omega_h::Few<Omega_h::Vector<2>, 2>& start_dest,
    const o::Few<o::Vector<2>, 2>& tri_edge) {
  // test_intersection();
  IntersectionResult intersection_point_struct =
      find_intersection_point(start_dest, tri_edge);
  if (intersection_point_struct.exists) {
    auto intersection_point = intersection_point_struct.point;
    return distance_between_points(start_dest[0], intersection_point);
  } else {
    return -1.0;
  }
}

OMEGA_H_DEVICE void search_through_mesh(const o::Mesh& mesh, o::Vector<3> x) {
  o::Vector<2> x_rz = {x[0], x[2]};
  auto coords = mesh.coords();
  auto n_faces = mesh.nfaces();
  auto face2nodes = mesh.get_adj(o::FACE, o::VERT).ab2b;
  bool found = false;
  for (int i = 0; i < n_faces; i++) {
    auto face_nodes = o::gather_verts<3>(face2nodes, i);
    Omega_h::Few<Omega_h::Vector<2>, 3> face_coords;
    face_coords = o::gather_vectors<3, 2>(coords, face_nodes);
    auto bcc = o::barycentric_from_global<2, 2>(x_rz, face_coords);
    if (all_positive(bcc)) {
      found = true;
      printf("Found the particle in element: %d\n", i);
      printf("Barycentric coordinates: %f, %f, %f\n", bcc[0], bcc[1], bcc[2]);
      printf("Coordinates of the face: \n");
      for (int j = 0; j < 3; j++) {
        printf("%.16f, %.16f\n", face_coords[j][0], face_coords[j][1]);
      }
      printf("Position of the particle was: %.16f, %.16f\n", x_rz[0], x_rz[1]);
    }
  }
  if (!found) {
    printf("Particle not found in the mesh\n");
  }
}

OMEGA_H_DEVICE IntersectionResult find_intersection_point(
    o::Few<o::Vector<2>, 2> line1, o::Few<o::Vector<2>, 2> line2) {
  IntersectionResult result;
  auto b = line2[0] - line1[0];
  o::Matrix<2, 2> A;
  A[0] = line1[1] - line1[0];
  A[1] = line2[0] - line2[1];
  // A = o::transpose(A);
  //  print the matrix
  // printf("Matrix A: \n");
  // for (int i = 0; i < 2; i++){
  //   for (int j = 0; j < 2; j++){
  //     printf("%f ", A[i][j]);
  //   }
  //   printf("\n");
  // }
  auto det = o::determinant(A);
  bool valid_det = (std::abs(det) > EPSILON);
  o::Vector<2> x = o::invert(A) * b;
  // if intersects near the origin, return the origin
  bool on_origin =
      (x[0] > -EPSILON * 100 && x[0] < 0 && valid_det);  // TODO: magic number
  result.exists = on_origin ? true : result.exists;
  result.point = on_origin ? line1[0] : result.point;

  bool intersects = (x[0] >= 0 && x[0] <= 1 && x[1] >= 0 && x[1] <= 1);

  o::Vector<2> intersection_point = (1 - x[0]) * line1[0] + x[0] * line1[1];
  result.exists =
      (intersects && !on_origin && valid_det) ? true : result.exists;
  result.point = (intersects && !on_origin && valid_det) ? intersection_point
                                                         : result.point;
  return result;
}

OMEGA_H_DEVICE bool counter_clockwise(const Omega_h::Vector<2>& a,
                                      const Omega_h::Vector<2>& b,
                                      const Omega_h::Vector<2>& c) {
  return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]);
}

// HACK to avoid having an unguarded comma in the PS PARALLEL macro
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors(o::Reals const& a,
                                             o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

OMEGA_H_DEVICE o::Vector<3> sampleRandomDirection(const double A,
                                                  random_pool_t random_pool) {
  // ref
  // https://docs.openmc.org/en/stable/methods/neutron_physics.html#isotropic-angular-distribution
  // ref
  // std::random_device rd;
  // std::mt19937 gen(0);
  // std::uniform_real_distribution<> dis(0, 1);
  auto gen = random_pool.get_state();
  double rn = gen.drand(0., 1.);
  double rn2 = gen.drand(0., 1.);
  random_pool.free_state(gen);
  double mu = 2 * rn - 1;
  // cosine in the particles incident direction
  double mu_lab = (1 + A * mu) / std::sqrt(1 + 2 * A * mu + A * A);
  // cosine with the plane of the collision
  double nu_lab = 2 * rn2 - 1;
  o::Vector<3> dir;

  // TODO: replace this dummy direction with the actual direction
  // actual direction needs the incident direction
  dir[0] = std::sqrt(1 - mu_lab * mu_lab) * std::cos(2 * M_PI * nu_lab);
  dir[1] = std::sqrt(1 - mu_lab * mu_lab) * std::sin(2 * M_PI * nu_lab);
  dir[2] = mu_lab;
  return dir;
}

OMEGA_H_DEVICE void get_tet_centroid(const o::LOs& cells2nodes, o::LO e,
                                     const o::Reals& nodes2coords,
                                     o::Few<o::Real, 3>& center) {
  auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
  auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
  center = average(cell_nodes2coords);
}

OMEGA_H_DEVICE void get_tri_centroid(const o::LOs& cells2nodes, o::LO e,
                                     const o::Reals& nodes2coords,
                                     o::Few<o::Real, 2>& center) {
  auto cell_nodes2nodes = o::gather_verts<3>(cells2nodes, o::LO(e));
  auto cell_nodes2coords =
      o::gather_vectors<3, 2>(nodes2coords, cell_nodes2nodes);
  center = average(cell_nodes2coords);
}

OMEGA_H_DEVICE void cylindrical2cartesian(const o::Vector<3> cyl,
                                          o::Vector<3>& cartesian) {
  OMEGA_H_CHECK(cyl.size() == 3);

  cartesian[0] = cyl[0] * std::cos(cyl[1]);
  cartesian[1] = cyl[0] * std::sin(cyl[1]);
  cartesian[2] = cyl[2];
}

OMEGA_H_DEVICE void cartesian2cylindrical(const o::Vector<3> cartesian,
                                          o::Vector<3>& cyl) {
  cyl[0] = std::sqrt(cartesian[0] * cartesian[0] + cartesian[1] * cartesian[1]);
  cyl[1] = std::atan2(cartesian[1], cartesian[0]);
  cyl[2] = cartesian[2];
}

OMEGA_H_DEVICE double random_path_length(double lambda, random_pool_t pool) {
  // ref:
  // https://docs.openmc.org/en/stable/methods/neutron_physics.html#sampling-distance-to-next-collision
  auto gen = pool.get_state();
  double rn = gen.drand(0., 1.);
  pool.free_state(gen);
  double l = -std::log(rn) * lambda;
  return l;
}

#endif