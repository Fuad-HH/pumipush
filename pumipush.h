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
struct IntersectionBccResult {
  bool exists = false;
  o::Vector<3> bcc = o::Vector<3>{0.0, 0.0, 0.0};
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
 * Populate the particles based on the area of the elements to distribute
 * uniformly
 */
int distributeParticlesBasesOnArea(const p::Mesh& picparts, PS::kkLidView ppe,
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
void pseudo2Dpush(PS* ptcls, double lambda, random_pool_t pool);

void push(PS* ptcls, int np, double lambda, random_pool_t pool);

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

  auto det = o::determinant(A);
  bool valid_det = (std::abs(det) > EPSILON);
  o::Vector<2> x = o::invert(A) * b;
  // if intersects near the origin, return the origin
  bool on_origin = (x[0] > -EPSILON && x[0] < EPSILON && valid_det);
  result.exists = on_origin ? false : result.exists;
  // result.point = on_origin ? line1[0] : result.point;

  bool intersects = (x[0] > 0.0 && x[0] <= 1 && x[1] > 0.0 && x[1] <= 1);

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
  double theta = gen.drand(0., 2. * M_PI);
  double phi = Kokkos::acos(2 * gen.drand(0., 1.) - 1);
  random_pool.free_state(gen);
  // double mu = 2 * rn - 1;
  //// cosine in the particles incident direction
  // double mu_lab = (1 + A * mu) / std::sqrt(1 + 2 * A * mu + A * A);
  //// cosine with the plane of the collision
  // double nu_lab = 2 * rn2 - 1;
  o::Vector<3> dir;

  // TODO: replace this dummy direction with the actual direction
  // actual direction needs the incident direction
  // dir[0] = std::sqrt(1 - mu_lab * mu_lab) * std::cos(2 * M_PI * nu_lab);
  // dir[1] = std::sqrt(1 - mu_lab * mu_lab) * std::sin(2 * M_PI * nu_lab);
  // dir[2] = mu_lab;
  dir[0] = Kokkos::sin(phi) * Kokkos::cos(theta);
  dir[1] = Kokkos::sin(phi) * Kokkos::sin(theta);
  dir[2] = Kokkos::cos(phi);
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
/*
OMEGA_H_DEVICE o::Vector<3> move_particle_accross_boundary(int pid, const
o::Vector<3>& origin_rThz, const o::Vector<3>& dest_rThz, o::Real cur_int_dist)
{ o::Vector<3> dest_xyz; cylindrical2cartesian(dest_rThz, dest_xyz);
    o::Vector<3> origin_xyz;
    cylindrical2cartesian(origin_rThz, origin_xyz);
    o::Vector<3> modified_rThz;

    // it's the new position but with the same theta
    o::Vector<3> dest_rThz_Th_plane = {dest_rThz[0], origin_rThz[1],
dest_rThz[2]}; o::Vector<3> dest_xyz_Th_plane;
    cylindrical2cartesian(dest_rThz_Th_plane, dest_xyz_Th_plane);

    o::Vector<3> direction_xyz = (dest_xyz - origin_xyz) / o::norm(dest_xyz -
origin_xyz);
    //printf("INFO: Particle %d direction: %.16f %.16f %.16f\n", pid,
direction_xyz[0], direction_xyz[1], direction_xyz[2]);

#ifdef DEBUG
    OMEGA_H_CHECK(std::abs(o::norm(direction_xyz) - 1.0) < EPSILON);
#endif

    o::Vector<3> direction_Th_plane = (dest_xyz_Th_plane - origin_xyz) /
o::norm(dest_xyz_Th_plane - origin_xyz);
    //printf("INFO: Particle %d direction in Th plane: %.16f %.16f %.16f\n",
pid, direction_Th_plane[0], direction_Th_plane[1], direction_Th_plane[2]);

#ifdef DEBUG
    OMEGA_H_CHECK(std::abs(o::norm(direction_Th_plane) - 1.0) < EPSILON);
#endif

    o::Real dot_product = o::inner_product(direction_xyz, direction_Th_plane);
    printf("INFO: Particle %d dot product: %.16f\n", pid, dot_product);
#ifdef DEBUG
    OMEGA_H_CHECK(std::abs(dot_product) < 1.0);
#endif

    o::Real intersection_distance_xyz = cur_int_dist / dot_product;
    printf("INFO: Particle %d intersection distance in xyz: %f\n", pid,
intersection_distance_xyz);

    dest_xyz = origin_xyz + (direction_xyz * intersection_distance_xyz);
    cartesian2cylindrical(dest_xyz, modified_rThz);

    //o::Vector<3> direction_rThz = (dest_rThz - origin_rThz) /
o::norm(dest_rThz - origin_rThz);
    //modified_rThz[0] = origin_rThz[0] + direction_rThz[0] * cur_int_dist;
    //modified_rThz[2] = origin_rThz[2] + direction_rThz[2] * cur_int_dist;

    //o::Real del_theta = dest_rThz[1] - origin_rThz[1];
    //o::Real r_fraction = (modified_rThz[0] - origin_rThz[0]) / (dest_rThz[0] -
origin_rThz[0]);
    //modified_rThz[1] = origin_rThz[1] + del_theta * r_fraction;
    printf("INFO: Particle %d updated destination position: %.16f %.16f
%.16f\n", pid, modified_rThz[0], modified_rThz[1], modified_rThz[2]);

    return modified_rThz;
}
*/

OMEGA_H_DEVICE o::Real get_dest_theta(const o::Vector<3>& origin_rThz,
                                      const o::Vector<3>& dest_rThz) {
  o::Vector<3> dest_xyz;
  cylindrical2cartesian(dest_rThz, dest_xyz);
  o::Vector<3> origin_xyz;
  cylindrical2cartesian(origin_rThz, origin_xyz);
  o::Vector<3> modified_rThz;

  o::Vector<3> direction_xyz =
      (dest_xyz - origin_xyz) / o::norm(dest_xyz - origin_xyz);
  o::Real theta = Kokkos::atan2(direction_xyz[0], direction_xyz[1]);

  return theta;
}

OMEGA_H_DEVICE o::Few<o::Vector<3>, 3> barycentric_basis(
    const o::Few<o::Vector<2>, 3> tri_verts) {
  o::Few<o::Vector<3>, 3> basis;
  for (int i = 0; i < 3; i++) {
    basis[0][i] = tri_verts[i][0];
    basis[1][i] = tri_verts[i][1];
    basis[2][i] = 1.0;
  }
  return basis;
}

OMEGA_H_DEVICE o::Vector<2> barycentric2real(
    const o::Few<o::Vector<2>, 3> tri_verts, const o::Vector<3> bary) {
  o::Few<o::Vector<3>, 3> basis = barycentric_basis(tri_verts);
  o::Vector<2> real_coords;
  // real_coords = basis * bary;
  real_coords[0] =
      basis[0][0] * bary[0] + basis[0][1] * bary[1] + basis[0][2] * bary[2];
  real_coords[1] =
      basis[1][0] * bary[0] + basis[1][1] * bary[1] + basis[1][2] * bary[2];
  return real_coords;
}

OMEGA_H_DEVICE IntersectionBccResult find_intersection_with_bcc(
    const o::Vector<3> origin_bcc, const o::Vector<3> dest_bcc, int edge) {
  o::Vector<3> bcc_vector = dest_bcc - origin_bcc;
  o::Real u = origin_bcc[edge] / (origin_bcc[edge] - dest_bcc[edge]);
  o::Vector<3> ray_intersect_bcc = origin_bcc + u * bcc_vector;
  int start_vertex[3]{1, 2, 0};
  o::Real s = -1.0 * (ray_intersect_bcc[start_vertex[edge]] - 1);
  IntersectionBccResult result;
  // printf("INFO: u: %.16f, s: %.16f\n", u, s);
  result.exists = (s > 0.0 && s <= 1.0) &&
                  (u > 0.0 && u <= 1.0);  // todo remove magic number
  result.bcc = ray_intersect_bcc;

  return result;
}

/**
 * @brief check if the origin is inside the bcc and not on the corner
 * @param origin_bcc
 * @details checks if bcc is all postive and only one can be 0 at a time
 */
OMEGA_H_DEVICE void check_origin_bcc_validity(const o::Vector<3> origin_bcc) {
  OMEGA_H_CHECK(all_positive(origin_bcc, EPSILON));
  o::LO n_edges = 0;
  for (int i = 0; i < 3; i++) {
    n_edges += (origin_bcc[i] < -EPSILON);
  }
  OMEGA_H_CHECK(n_edges <= 1);
}

OMEGA_H_DEVICE o::LO get_edge_holding_point(const o::Vector<3> bcc,
                                            o::Few<o::LO, 3> el_edges) {
  o::LO edge = -1;
  for (int i = 0; i < 3; i++) {
    edge = (std::abs(bcc[i]) < EPSILON) ? el_edges[(i + 1) % 3] : edge;
  }
  return edge;
}

OMEGA_H_DEVICE o::LO get_the_other_adj_face_of_edge(
    o::LO edge, o::LO current_el, const o::LOs edge2faceOffsets,
    const o::LOs edge2faceFace, const o::Read<o::I8> exposed_edges) {
  o::LO other_face = -1;
  if (edge != -1) {
    // get the other face of the edge if it's on an edge
    o::LO n_adj_faces = edge2faceOffsets[edge + 1] - edge2faceOffsets[edge];
    bool on_boundary = (n_adj_faces == 1);
    OMEGA_H_CHECK(on_boundary == exposed_edges[edge]);
    if (!on_boundary) {
      other_face = (edge2faceFace[edge2faceOffsets[edge]] == current_el)
                       ? edge2faceFace[edge2faceOffsets[edge] + 1]
                       : edge2faceFace[edge2faceOffsets[edge]];
    }
  }
  return other_face;
}

/**
 * Get triangle area
 */
OMEGA_H_DEVICE o::Real area_tri(o::Few<o::Vector<2>, 3> tri_verts) {
  o::Few<o::Vector<2>, 2> basis22 = {tri_verts[1] - tri_verts[0],
                                     tri_verts[2] - tri_verts[0]};
  auto area = o::triangle_area_from_basis(basis22);
  return area;
}

/**
 * Get total area of a 2D mesh
 */
o::Real area_of_2d_mesh(o::Mesh& mesh);

/**
 * Set initial particle positions uniformly distributed in the parent element
 * and each element gets particles proportional to the area
 */
void setUniformPtclCoords(p::Mesh& picparts, PS* ptcls,
                          random_pool_t random_pool);

#endif