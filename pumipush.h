#ifndef PUMIPUSH_H
#define PUMIPUSH_H

#include <Omega_h_macros.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
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

#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_ptcl_ops.hpp"
// #include <team_policy.hpp>
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

// ******************** Function Prototypes ******************** //
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors(o::Reals const& a,
                                             o::Few<o::LO, 4> v);

/**
 * @brief Generate a random path length using an exponential distribution
 * TODO: make this lambda a mean free path length
 */
inline double random_path_length(double lambda);

o::Mesh readMesh(std::string meshFile, o::Library& lib);

/**
 * Populate the particles equally to all elements
 */
int distributeParticlesEqually(const p::Mesh& picparts, PS::kkLidView ppe,
                               const int numPtcls);

/**
 * Set the initial particle coordinates to the centroid of the parent element
 */
void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls);

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
inline o::Vector<3> sampleRandomDirection(const double A = 1.0);

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

#endif