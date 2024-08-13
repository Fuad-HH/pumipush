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

int main(int argc, char** argv) {
  test_intersection();
  test_on_edge_origin_case();
  auto lib = o::Library(&argc, &argv);
  OMEGA_H_CHECK(std::string(lib.version()) == OMEGA_H_SEMVER);
  bbox_varification_7k(&lib);
  bbox_verification_coarseWest(&lib);
  // checkCPNFileReading(&lib);
  bbox_varification_2dBox(&lib);
  check_cyl2cart();
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
}