// Gmsh project created on Fri Jun 14 13:01:37 2024
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {10, 0, 0, 1.0};
//+
Point(3) = {-10, 0, 0, 1.0};
//+
Point(4) = {0, 10, 0, 1.0};
//+
Point(5) = {0, -10, 0, 1.0};
//+
//+
Circle(1) = {4, 1, 5};
//+
Circle(2) = {5, 1, 2};
Circle(3) = {2, 1, 4};
//+
//+
Point(6) = {0, -4, 0, 1.0};
//+
Point(7) = {0, 4, 0, 1.0};
//+
Circle(4) = {7, 1, 6};
//+

Line(5) = {1, 6};
Line(6) = {1, 7};
Line(7) = {7, 4};
Line(8) = {6, 5};
Line(9) = {1, 2};



//+
Curve Loop(1) = {3, -7, -6, 9};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, -9, 5, 8};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {4, -5, 6};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {1, -8, -4, 7};
//+
Plane Surface(4) = {4};
//+
Extrude {0, 0, 10} {
  Surface{4}; Surface{3}; Surface{1}; Surface{2}; Point{2}; Point{4}; Point{3}; Point{5}; Point{6}; Point{1}; Point{7}; Curve{1}; Curve{4}; Curve{6}; Curve{7}; Curve{9}; Curve{5}; Curve{8}; Curve{2}; Curve{3};
}
//+
Physical Surface(94) = {26};
//+
Physical Surface(95) = {30};
//+
Physical Surface(96) = {47};
//+
Physical Surface(97) = {43};
//+
Physical Surface(98) = {22};
//+
Physical Surface(99) = {18};
//+
Physical Surface(100) = {79};
//+
Physical Surface(101) = {57};
//+
Physical Surface(102) = {69};
//+

//+
Physical Volume(103) = {1};
//+
Physical Volume(104) = {2};
//+
Physical Volume(105) = {4};
//+
Physical Volume(106) = {3};
