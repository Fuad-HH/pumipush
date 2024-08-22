#include "pumipush.h"

#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_tag.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

bool is_close(double a, double b, double tol = 1e-5) {
  return std::abs(a - b) < tol;
}
template <typename T>
bool match_with_expected(T min, T max, T expected_min, T expected_max);
void bbox_varification_7k(o::Library* lib);
void bbox_verification_coarseWest(o::Library* lib);
void checkCPNFileReading(o::Library* lib);
void bbox_varification_2dBox(o::Library* lib);
void check_cyl2cart();
void test_intersection();
void test_on_edge_origin_case();
void test_move_particle_accross_element_boundary();
void test_bcc_intersection_methods();
void test_tri_node_edge_adj_info_consistency(o::Library* lib);
void test_relation_between_bcc_and_edge(o::Library* lib);
void test_bcc_intersection_for_p3d();
void test_intersection_LCPP_10627();
void test_intersect_all_edges();
void test_element_case_9834();

int main(int argc, char** argv) {
  auto lib = o::Library(&argc, &argv);
  test_element_case_9834();
  test_intersect_all_edges();
  test_intersection_LCPP_10627();
  // test_bcc_intersection_for_p3d();
  test_relation_between_bcc_and_edge(&lib);
  test_tri_node_edge_adj_info_consistency(&lib);
  test_bcc_intersection_methods();
  test_intersection();
  // test_on_edge_origin_case();
  OMEGA_H_CHECK(std::string(lib.version()) == OMEGA_H_SEMVER);
  bbox_varification_7k(&lib);
  bbox_verification_coarseWest(&lib);
  // checkCPNFileReading(&lib);
  bbox_varification_2dBox(&lib);
  check_cyl2cart();
  test_move_particle_accross_element_boundary();
}

void test_element_case_9834() {
  printf("Test: element case 9834...\n");
  o::Few<o::Vector<2>, 3> tri = {{2.7071293727351278, -0.0930694021069807},
                                 {2.6999306350252650, -0.0994799075658618},
                                 {2.7034180027260302, -0.1018031881758616}};
  o::Vector<2> origin = {2.7015263010937849, -0.1005429385838452};
  o::Vector<2> dest = {2.7015264483607950, -0.1005447430684491};

  auto origin_bcc = o::barycentric_from_global<2, 2>(origin, tri);
  auto dest_bcc = o::barycentric_from_global<2, 2>(dest, tri);
  printf("BCC-origin: %f %f %f\n", origin_bcc[0], origin_bcc[1], origin_bcc[2]);
  OMEGA_H_CHECK(all_positive(origin_bcc, 0.0));
  OMEGA_H_CHECK(!all_positive(dest_bcc, 0.0));

  IntersectionBccResult result_e0 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 0);
  IntersectionBccResult result_e1 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 1);
  IntersectionBccResult result_e2 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 2);
  printf("Exists: %d %d %d\n", result_e0.exists, result_e1.exists,
         result_e2.exists);
}
void test_intersect_all_edges() {
  printf("Test: intersect all edges...\n");
  o::Few<o::Vector<2>, 3> tri = {{1.86603, 0.5}, {2.13397, -0.5}, {1, 0}};
  o::Vector<2> origin = {1.9446295144898407, 0.1260601696803720};
  o::Vector<2> dest = {2.0007231239607082, 0.1512722036160794};

  auto origin_bcc = o::barycentric_from_global<2, 2>(origin, tri);
  auto dest_bcc = o::barycentric_from_global<2, 2>(dest, tri);
  OMEGA_H_CHECK(all_positive(origin_bcc, 0.0));
  OMEGA_H_CHECK(!all_positive(dest_bcc, 0.0));

  IntersectionBccResult result_e0 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 0);
  IntersectionBccResult result_e1 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 1);
  IntersectionBccResult result_e2 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 2);
  printf("Exists: %d %d %d\n", result_e0.exists, result_e1.exists,
         result_e2.exists);
}

void test_intersection_LCPP_10627() {
  printf("Test: intersection for LCPP-10627...\n");
  o::Few<o::Vector<2>, 3> el10627 = {{2.3638890457070847, 0.3339009278951910},
                                     {2.3614440013667641, 0.3265147585787270},
                                     {2.3669213202267083, 0.3291142310482917}};
  o::Vector<2> dest{2.3630059791071032, 0.3312355204866098};
  o::Vector<2> origin = (el10627[1] + el10627[2]) / 2.0;

  auto origin_bcc = o::barycentric_from_global<2, 2>(origin, el10627);
  auto dest_bcc = o::barycentric_from_global<2, 2>(dest, el10627);

  printf("BCC-origin: %f %f %f\n", origin_bcc[0], origin_bcc[1], origin_bcc[2]);
  printf("BCC-dest: %f %f %f\n", dest_bcc[0], dest_bcc[1], dest_bcc[2]);
  OMEGA_H_CHECK(all_positive(origin_bcc, EPSILON));
  OMEGA_H_CHECK(!all_positive(dest_bcc, 0.0));

  IntersectionBccResult result_e2 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 2);
  OMEGA_H_CHECK(result_e2.exists);
}
void test_bcc_intersection_for_p3d() {
  printf("Test: BCC intersection for pseudo3D problem...\n");
  o::Few<o::Vector<2>, 3> tri = {{1, -1}, {1.1397, -0.5}, {1, 0}};
  o::Vector<2> origin = {1.377992, -0.5};
  o::Vector<2> dest = {3, 1};

  auto origin_bcc = o::barycentric_from_global<2, 2>(origin, tri);
  auto dest_bcc = o::barycentric_from_global<2, 2>(dest, tri);

  IntersectionBccResult result_e0 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 0);
  IntersectionBccResult result_e1 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 1);
  IntersectionBccResult result_e2 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 2);
  printf("BCC: %f %f %f\n", origin_bcc[0], origin_bcc[1], origin_bcc[2]);

  OMEGA_H_CHECK(result_e0.exists);
  OMEGA_H_CHECK(!result_e1.exists);
  OMEGA_H_CHECK(!result_e2.exists);
}

void test_relation_between_bcc_and_edge(o::Library* lib) {
  printf("Test: relation between bcc and edge...\n");
  o::Mesh mesh(lib);
  o::binary::read("assets/square.osh", lib->world(), &mesh);
  OMEGA_H_CHECK(mesh.dim() == 2);
  const auto face2edgeEdge = mesh.ask_down(o::FACE, o::EDGE).ab2b;
  const auto face2nodeNode = mesh.ask_down(o::FACE, o::VERT).ab2b;
  const auto edge2nodeNode = mesh.ask_down(o::EDGE, o::VERT).ab2b;
  const auto coords = mesh.coords();

  auto verify_bcc_edge_relation = OMEGA_H_LAMBDA(o::LO id) {
    o::LO face_id = 0;
    o::Vector<2> origin = {3.0, 0.5};
    o::Few<o::LO, 3> face_nodes = {face2nodeNode[3 * face_id],
                                   face2nodeNode[3 * face_id + 1],
                                   face2nodeNode[3 * face_id + 2]};
    o::Few<o::LO, 3> face_edges = {face2edgeEdge[3 * face_id],
                                   face2edgeEdge[3 * face_id + 1],
                                   face2edgeEdge[3 * face_id + 2]};
    auto face_node_coords = o::gather_vectors<3, 2>(coords, face_nodes);
    auto bcc = o::barycentric_from_global<2, 2>(origin, face_node_coords);
    // printf("BCC: %f %f %f\n", bcc[0], bcc[1], bcc[2]);
    o::LO origin_on_edge = -1;
    o::LO cor_node = -1;
    for (int i = 0; i < 3; i++) {
      bool is_on_edge = std::abs(bcc[i]) < EPSILON;
      cor_node = (is_on_edge) ? face_nodes[i] : cor_node;
      origin_on_edge = (is_on_edge) ? face_edges[(i + 1) % 3] : origin_on_edge;
    }
    OMEGA_H_CHECK(origin_on_edge == 17 && cor_node == 13);
  };
  o::parallel_for("verify_bcc_edge_relation", 1, verify_bcc_edge_relation);
}

void test_tri_node_edge_adj_info_consistency(o::Library* lib) {
  // load  a mesh
  Omega_h::Mesh mesh(lib);
  Omega_h::binary::read("assets/coarseWestLCPP.osh", lib->world(), &mesh);
  OMEGA_H_CHECK(mesh.dim() == 2);
  const auto face2edgeEdge = mesh.ask_down(o::FACE, o::EDGE).ab2b;
  const auto face2nodeNode = mesh.ask_down(o::FACE, o::VERT).ab2b;
  const auto edge2nodeNode = mesh.ask_down(o::EDGE, o::VERT).ab2b;

  // get the nodes of the 0th face
  auto verify_edge_node_relation = OMEGA_H_LAMBDA(o::LO face_id) {
    // o::LO face_id = 9;
    // printf("Checking Face %d...\n", face_id);
    o::Few<o::LO, 3> face_nodes = {face2nodeNode[3 * face_id],
                                   face2nodeNode[3 * face_id + 1],
                                   face2nodeNode[3 * face_id + 2]};
    o::Few<o::LO, 3> face_edges = {face2edgeEdge[3 * face_id],
                                   face2edgeEdge[3 * face_id + 1],
                                   face2edgeEdge[3 * face_id + 2]};
    // printf("Face nodes: %d %d %d\n", face_nodes[0], face_nodes[1],
    //        face_nodes[2]);
    // printf("Face edges: %d %d %d\n", face_edges[0], face_edges[1],
    //        face_edges[2]);
    o::Few<o::Few<o::LO, 2>, 3> edge_nodes;
    for (int local_edge_id = 0; local_edge_id < 3; local_edge_id++) {
      edge_nodes[local_edge_id] = {
          edge2nodeNode[2 * face_edges[local_edge_id]],
          edge2nodeNode[2 * face_edges[local_edge_id] + 1]};
      OMEGA_H_CHECK(edge_nodes[local_edge_id][0] !=
                    face_nodes[(local_edge_id + 2) % 3]);
      OMEGA_H_CHECK(edge_nodes[local_edge_id][1] !=
                    face_nodes[(local_edge_id + 2) % 3]);
      // printf("Edge %d nodes: %d %d\n", face_edges[local_edge_id],
      // edge_nodes[local_edge_id][0], edge_nodes[local_edge_id][1]);
    }
  };
  o::parallel_for("verify_edge_node_relation", mesh.nelems(),
                  verify_edge_node_relation);
}

void test_bcc_intersection_methods() {
  printf("Test: BCC intersection methods...\n");
  o::Few<o::Vector<2>, 3> tri_1 = {{0, 0}, {1, 0}, {0.5, 1}};
  auto basis_1 = barycentric_basis(tri_1);
  o::Few<o::Vector<3>, 3> expected_basis = {{0, 1, 0.5}, {0, 0, 1}, {1, 1, 1}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      OMEGA_H_CHECK(std::abs(basis_1[i][j] - expected_basis[i][j]) < 1.0e-6);
    }
  }

  o::Few<o::Vector<2>, 3> tri_2 = {{1, 2}, {3, 4}, {5, 6}};
  auto basis_2 = barycentric_basis(tri_2);
  expected_basis = {{1, 3, 5}, {2, 4, 6}, {1, 1, 1}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      OMEGA_H_CHECK(std::abs(basis_2[i][j] - expected_basis[i][j]) < 1.0e-6);
    }
  }

  o::Vector<3> bc3 = {0, 0, 1};
  auto c1 = barycentric2real(tri_2, bc3);
  // expected {5, 6}
  OMEGA_H_CHECK(std::abs(c1[0] - tri_2[2][0]) < 1.0e-6 &&
                std::abs(c1[1] - tri_2[2][1]) < 1.0e-6);
  bc3 = {0, 1, 0};
  c1 = barycentric2real(tri_2, bc3);
  OMEGA_H_CHECK(std::abs(c1[0] - tri_2[1][0]) < 1.0e-6 &&
                std::abs(c1[1] - tri_2[1][1]) < 1.0e-6);
  bc3 = {0, 0.5, 0.5};
  c1 = barycentric2real(tri_2, bc3);
  OMEGA_H_CHECK(std::abs(c1[0] - 4) < 1.0e-6 && std::abs(c1[1] - 5) < 1.0e-6);

  o::Few<o::Vector<2>, 2> ray = {{0.2, 0.2}, {-1, 0.5}};
  auto origin_bcc = o::barycentric_from_global<2, 2>(ray[0], tri_1);
  auto dest_bcc = o::barycentric_from_global<2, 2>(ray[1], tri_1);
  IntersectionBccResult result_e0 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 0);
  IntersectionBccResult result_e1 =
      find_intersection_with_bcc(origin_bcc, dest_bcc, 1);

  OMEGA_H_CHECK(!result_e0.exists);
  OMEGA_H_CHECK(result_e1.exists);
  o::Vector<3> expected_intersection_bcc = {0.777777777777, 0.0,
                                            0.222222222222};
  for (int i = 0; i < 3; i++) {
    OMEGA_H_CHECK(std::abs(result_e1.bcc[i] - expected_intersection_bcc[i]) <
                  1.0e-6);
  }

  o::Few<o::Vector<2>, 3> u_tri{{-2.0, -3}, {7, -1.0}, {-2, 5}};
  o::Few<o::Vector<2>, 2> ray_inside2out{{0, 0}, {-23, 10}};
  origin_bcc = o::barycentric_from_global<2, 2>(ray_inside2out[0], u_tri);
  OMEGA_H_CHECK(all_positive(origin_bcc, 0.0));
  dest_bcc = o::barycentric_from_global<2, 2>(ray_inside2out[1], u_tri);
  OMEGA_H_CHECK(!all_positive(dest_bcc, 0.0));
  result_e1 = find_intersection_with_bcc(origin_bcc, dest_bcc, 1);
  OMEGA_H_CHECK(result_e1.exists);
  auto intersection_point = barycentric2real(u_tri, result_e1.bcc);
  OMEGA_H_CHECK(std::abs(intersection_point[0] + 2.0) < 1.0e-6);
  OMEGA_H_CHECK(intersection_point[1] > -3.0 && intersection_point[1] < 5.0);

  result_e0 = find_intersection_with_bcc(origin_bcc, dest_bcc, 0);
  auto result_e2 = find_intersection_with_bcc(origin_bcc, dest_bcc, 2);
  OMEGA_H_CHECK(!result_e0.exists);
  OMEGA_H_CHECK(!result_e2.exists);

  o::Few<o::Vector<2>, 2> ray_out2in{{-23, 10}, {-1, 0.1}};
  origin_bcc = o::barycentric_from_global<2, 2>(ray_out2in[0], u_tri);
  OMEGA_H_CHECK(!all_positive(origin_bcc, 0.0));
  dest_bcc = o::barycentric_from_global<2, 2>(ray_out2in[1], u_tri);
  OMEGA_H_CHECK(all_positive(dest_bcc, 0.0));
  result_e1 = find_intersection_with_bcc(origin_bcc, dest_bcc, 1);
  OMEGA_H_CHECK(result_e1.exists);
  intersection_point = barycentric2real(u_tri, result_e1.bcc);
  OMEGA_H_CHECK(std::abs(intersection_point[0] + 2.0) < 1.0e-6);
  OMEGA_H_CHECK(intersection_point[1] > -3.0 && intersection_point[1] < 5.0);

  result_e0 = find_intersection_with_bcc(origin_bcc, dest_bcc, 0);
  result_e2 = find_intersection_with_bcc(origin_bcc, dest_bcc, 2);
  OMEGA_H_CHECK(!result_e0.exists);
  OMEGA_H_CHECK(!result_e2.exists);

  o::Few<o::Vector<2>, 2> ray_in2in{{-1, 0.1}, {0, 0}};
  origin_bcc = o::barycentric_from_global<2, 2>(ray_in2in[0], u_tri);
  OMEGA_H_CHECK(all_positive(origin_bcc, 0.0));
  dest_bcc = o::barycentric_from_global<2, 2>(ray_in2in[1], u_tri);
  OMEGA_H_CHECK(all_positive(dest_bcc, 0.0));
  result_e1 = find_intersection_with_bcc(origin_bcc, dest_bcc, 1);
  OMEGA_H_CHECK(!result_e1.exists);
  result_e0 = find_intersection_with_bcc(origin_bcc, dest_bcc, 0);
  result_e2 = find_intersection_with_bcc(origin_bcc, dest_bcc, 2);
  OMEGA_H_CHECK(!result_e0.exists);

  printf("BCC intersection methods test passed\n");
}

void check_is_inside_tet() {}

void bbox_varification_7k(o::Library* lib) {
  printf("Bounding box verification for 7k mesh (OH repo)\n");
  o::Vector<2> min, max;
  Omega_h::Mesh mesh7kcube(lib);
  Omega_h::binary::read("assets/7k.osh", lib->world(), &mesh7kcube);
  std::cout << "Mesh loaded successfully with " << mesh7kcube.nelems()
            << " elements\n\t\t\t"
            << "and " << mesh7kcube.nverts() << " vertices\n";
  get_bounding_box_in_xy_plane(mesh7kcube, min, max);
  o::Vector<2> expected_min_7k = {-0.5, -0.5};
  o::Vector<2> expected_max_7k = {65, 65};
  OMEGA_H_CHECK(
      match_with_expected(min, max, expected_min_7k, expected_max_7k));
}

void bbox_verification_coarseWest(o::Library* lib) {
  printf("TEST: Bounding box verification for coarseWestLCPP mesh\n");
  o::Vector<2> min, max;
  Omega_h::Mesh meshwestcoarse(lib);
  Omega_h::binary::read("assets/coarseWestLCPP.osh", lib->world(),
                        &meshwestcoarse);
  std::cout << "Mesh loaded successfully with " << meshwestcoarse.nelems()
            << " elements\n\t\t\t"
            << "and " << meshwestcoarse.nverts() << " vertices"
            << "dim = " << meshwestcoarse.dim() << "\n";
  get_bounding_box_in_xy_plane(meshwestcoarse, min, max);
  o::Vector<2> expected_min_west = {1.83431, -0.94};
  o::Vector<2> expected_max_west = {3.19847, 0.7986};
  OMEGA_H_CHECK(
      match_with_expected(min, max, expected_min_west, expected_max_west));
}

void checkCPNFileReading(o::Library* lib) {
  printf("TEST: Checking CPN file reading\n");
  std::string cpn_file = "test.cpn";
  o::Write<o::LO> owners(7, 0);
  ownerFromCPN(cpn_file, owners);

  std::vector<o::LO> expected_owner = {0, 0, 1, 2, 3, 3, 4};
  OMEGA_H_CHECK(owners.size() == expected_owner.size());
  for (int i = 0; i < owners.size(); i++) {
    OMEGA_H_CHECK(owners[i] == expected_owner[i]);
  }
}

void bbox_varification_2dBox(o::Library* lib) {
  printf("TEST: Bounding box verification for 2d box mesh\n");
  o::Vector<2> min, max;
  // 2d case
  Omega_h::Mesh mesh2d(lib);
  Omega_h::binary::read("assets/2d_box.osh", lib->world(), &mesh2d);
  std::cout << "Mesh loaded successfully with " << mesh2d.nelems()
            << " elements\n\t\t\t"
            << "and " << mesh2d.nverts() << " vertices\n";
  get_bounding_box_in_xy_plane(mesh2d, min, max);
  o::Vector<2> expected_min_2d = {0, 0};
  o::Vector<2> expected_max_2d = {1, 0.8};
  OMEGA_H_CHECK(
      match_with_expected(min, max, expected_min_2d, expected_max_2d));
}

template <typename T>
bool match_with_expected(T min, T max, T expected_min, T expected_max) {
  if (!is_close(min[0], expected_min[0]) ||
      !is_close(min[1], expected_min[1]) ||
      !is_close(max[0], expected_max[0]) ||
      !is_close(max[1], expected_max[1])) {
    return false;
  }
  return true;
}

void check_cyl2cart() {
  o::Vector<3> cyl = {1, 0, 0};
  o::Vector<3> cart;
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], 1));
  OMEGA_H_CHECK(is_close(cart[1], 0));
  OMEGA_H_CHECK(is_close(cart[2], 0));

  cyl = {1, o::PI / 2, 3};
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], 0));
  OMEGA_H_CHECK(is_close(cart[1], 1));
  OMEGA_H_CHECK(is_close(cart[2], 3));

  cyl = {1, o::PI, 0};
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], -1));
  OMEGA_H_CHECK(is_close(cart[1], 0));
  OMEGA_H_CHECK(is_close(cart[2], 0));

  cyl = {2, -o::PI / 2, 0};
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], 0));
  OMEGA_H_CHECK(is_close(cart[1], -2));
  OMEGA_H_CHECK(is_close(cart[2], 0));
}

void cart2cyl() {
  o::Vector<3> cart = {1, 0, 0};
  o::Vector<3> cyl;
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 1));
  OMEGA_H_CHECK(is_close(cyl[1], 0));
  OMEGA_H_CHECK(is_close(cyl[2], 0));

  cart = {0, 1, 3};
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 1));
  OMEGA_H_CHECK(is_close(cyl[1], o::PI / 2));
  OMEGA_H_CHECK(is_close(cyl[2], 3));

  cart = {-1, 0, 0};
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 1));
  OMEGA_H_CHECK(is_close(cyl[1], o::PI));
  OMEGA_H_CHECK(is_close(cyl[2], 0));

  // 4th quadrant
  cart = {0, -2, 5};
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 2));
  OMEGA_H_CHECK(is_close(cyl[1], -o::PI / 2));
  OMEGA_H_CHECK(is_close(cyl[2], 5));
}
void test_on_edge_origin_case() {
  printf("Test: origin on edge...\n");
  o::Few<o::Vector<2>, 2> edge = {{2.6217217103259052, 0.2984345434763631},
                                  {2.6279145475054091, 0.2928020473794012}};
  o::Few<o::Vector<2>, 2> ray = {{2.6231560255680102, 0.2971300081293064},
                                 {2.6228101864056441, 0.2980303188213828}};
  auto intersection_point = find_intersection_point(ray, edge);
  OMEGA_H_CHECK(intersection_point.exists);
  double intersection_distance = find_intersection_distance_tri(ray, edge);
  OMEGA_H_CHECK(std::abs(intersection_distance - 0.000000) < 1.0e-6);
}
void test_intersection() {
  printf("Test: intersection...\n");
  o::Few<o::Vector<2>, 2> line1 = {{-1, 0.5}, {1, 0.5}};
  o::Few<o::Vector<2>, 2> line2 = {{0, 1}, {0, 2}};
  auto intersection_point = find_intersection_point(line1, line2);
  OMEGA_H_CHECK(!intersection_point.exists);
  double dist = distance_between_points(line1[0], line1[1]);
  OMEGA_H_CHECK(is_close(dist, 2.0));
  dist = distance_between_points(line2[0], line2[1]);
  OMEGA_H_CHECK(is_close(dist, 1.0));

  line1 = {{1.8, 2.1}, {0.8, 1.1}};
  line2 = {{1, 1.25}, {0, 1.25}};

  dist = distance_between_points(line1[0], line1[1]);
  OMEGA_H_CHECK(is_close(dist, std::sqrt(2.0)));
  dist = distance_between_points(line2[0], line2[1]);
  OMEGA_H_CHECK(is_close(dist, 1.0));

  intersection_point = find_intersection_point(line1, line2);
  o::Vector<2> expected_intersection_point = {0.95, 1.25};
  OMEGA_H_CHECK(intersection_point.exists);
  // print the intersection point
  // printf("Intersection point: %f %f\n", intersection_point.value()[0],
  // intersection_point.value()[1]);
  OMEGA_H_CHECK(o::are_close(intersection_point.point,
                             expected_intersection_point, 1.0e-10));
  double distance = find_intersection_distance_tri(line1, line2);
  dist = distance_between_points(line1[0], intersection_point.point);
  printf("Distance - direct: %f\n", distance);

  printf("Distance: %f\n", distance);
  OMEGA_H_CHECK(std::abs(distance - dist) < 1.0e-6);
  OMEGA_H_CHECK(std::abs(distance - 1.202082) < 1.0e-6);

  line1 = {{2.562371, 0.110059}, {2.562387, 0.110075}};
  line2 = {{2.563669, 0.111339}, {2.557894, 0.105644}};
  distance = find_intersection_distance_tri(line1, line2);
  printf("Distance: %f\n", distance);

  o::Few<o::Vector<2>, 3> face31_coords = {
      {2.568239, 0.102393}, {2.563669, 0.111339}, {2.557894, 0.105644}};
  o::Few<o::Vector<2>, 3> face38_coords = {
      {2.55291, 0.114357}, {2.563669, 0.111339}, {2.557894, 0.105644}};

  auto bcc =
      o::barycentric_from_global<2, 2>({2.562387, 0.110075}, face31_coords);
  printf("Barycentric for 31: %f %f %f\n", bcc[0], bcc[1], bcc[2]);
  OMEGA_H_CHECK(!all_positive(bcc));
  bcc = o::barycentric_from_global<2, 2>({2.562387, 0.110075}, face38_coords);
  printf("Barycentric for 38: %f %f %f\n", bcc[0], bcc[1], bcc[2]);
  OMEGA_H_CHECK(all_positive(bcc));

  // gpu intersection case
  line1 = {{2.6220084679273516, 0.4999999999986200},
           {73.7249693813762264, 71.5342561928560059}};
  line2 = {{3.0000000000000000, -0.0000000000055056},
           {3.0000000000000000, 0.9999999999972258}};
  distance = find_intersection_distance_tri(line1, line2);
  o::Real expected_distance = 0.534303;
  OMEGA_H_CHECK(std::abs(distance - expected_distance) < 1.0e-6);
}

void test_move_particle_accross_element_boundary() {
  int pid = 0;
  o::Vector<3> origin_rThz{2.6220084679273516, 2.0175472867126008,
                           0.4999999999986200};
  o::Vector<3> dest_rThz{73.7249693813762264, 1.3969918345852945,
                         71.5342561928560059};
  o::Vector<3> edgevert1{3.0000000000000000, -0.0000000000055056};
  o::Vector<3> edgevert2{3.0000000000000000, 0.9999999999972258};

  o::Few<o::Vector<2>, 2> ray{{origin_rThz[0], origin_rThz[2]},
                              {dest_rThz[0], dest_rThz[2]}};
  o::Few<o::Vector<2>, 2> edge{{edgevert1[0], edgevert1[1]},
                               {edgevert2[0], edgevert2[1]}};

  auto intersection_point = find_intersection_point(ray, edge);
  OMEGA_H_CHECK(intersection_point.exists);
  OMEGA_H_CHECK(std::abs(intersection_point.point[0] - 3.0) < 1.0e-6);
  o::Real dt_c = find_intersection_distance_tri(ray, edge);
  o::Real dt = 0.534303;
  OMEGA_H_CHECK(std::abs(dt - dt_c) < 1.0e-6);
  // o::Vector<3> new_rThz = move_particle_accross_boundary(pid, origin_rThz,
  // dest_rThz, dt_c); o::Vector<2> expected_rz = intersection_point.point;
  // OMEGA_H_CHECK(o::are_close({new_rThz[0], new_rThz[2]},
  // expected_rz, 1.0e-6));
}