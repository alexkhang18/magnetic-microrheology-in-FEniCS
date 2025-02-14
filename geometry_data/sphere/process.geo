Merge "sphere_downsampled.stl";

CreateTopology;

Physical Surface (1) = {1}; // sphere

Surface Loop (1) = {1}; // sphere
Volume (1) = {1};

Physical Volume (200) = {1}; // sphere

// Lengths
l = 150.;
length = l;
height = l;
depth = l;
lcar1 = 0.5;

centroidx = 0;
centroidy = 0;
centroidz = 0;

// Translate {length, height, depth} {Surface{1};}

// Outer Boundary Points
Point(1) = {centroidx+length/2,centroidy+height/2,centroidz+depth/2,lcar1}; 
Point(2) = {centroidx+length/2,centroidy+height/2,centroidz-depth/2,lcar1}; 
Point(3) = {centroidx-length/2,centroidy+height/2,centroidz+depth/2,lcar1}; 
Point(4) = {centroidx-length/2,centroidy-height/2,centroidz+depth/2,lcar1}; 
Point(5) = {centroidx+length/2,centroidy-height/2,centroidz+depth/2,lcar1}; 
Point(6) = {centroidx+length/2,centroidy-height/2,centroidz-depth/2,lcar1}; 
Point(7) = {centroidx-length/2,centroidy+height/2,centroidz-depth/2,lcar1}; 
Point(8) = {centroidx-length/2,centroidy-height/2,centroidz-depth/2,lcar1}; 


Line(1) = {3,1};
Line(2) = {3,7};
Line(3) = {7,2};
Line(4) = {2,1};
Line(5) = {1,5};
Line(6) = {5,4};
Line(7) = {4,8};
Line(8) = {8,6};
Line(9) = {6,5};
Line(10) = {6,2};
Line(11) = {3,4};
Line(12) = {8,7};
Line Loop(13) = {-6,-5,-1,11};
Plane Surface(14) = {13};
Line Loop(15) = {4,5,-9,10};
Plane Surface(16) = {15};
Line Loop(17) = {-3,-12,8,10};
Plane Surface(18) = {17};
Line Loop(19) = {7,12,-2,11};
Plane Surface(20) = {19};
Line Loop(21) = {-4,-3,-2,1};
Plane Surface(22) = {21};
Line Loop(23) = {8,9,6,7};
Plane Surface(24) = {23};


Surface Loop(25) = {1,14,24,-18,22,16,-20}; // gel
Volume(26) = {25}; // gel

Physical Surface(101) = {14,16,18,20,22,24}; // gel
Physical Volume(100) = {26}; // gel


Mesh.CharacteristicLengthFactor = 0.5;
// Characteristic Length{PointsOf{Physical Surface{101};}} = 20;
Characteristic Length{PointsOf{Physical Surface{101};}} = 40;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "process.msh";
