import os
import glob
import xml.etree.ElementTree as ET

# Directory containing the .pvtu files
directory = "*"

# Find all pieces.pvtu files
pvtu_files = glob.glob(os.path.join(directory, "**", "pieces.pvtu"), recursive=True)

# Create the root of the .pvd file
root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian", compressor="vtkZLibDataCompressor")
collection = ET.SubElement(root, "Collection")

# Sort the pvtu files by time step (assumed to be in the directory name)
for pvtu_file in sorted(pvtu_files):
    # Extract time step from the directory name (assuming the format pseudoMC_tXX_r0.vtk)
    dir_name = os.path.basename(os.path.dirname(pvtu_file))
    timestep = int(dir_name.split('_t')[1].split('_')[0])

    # Add the DataSet element
    ET.SubElement(collection, "DataSet", timestep=str(timestep), group="", part="0", file=os.path.relpath(pvtu_file, directory))

# Write the tree to an XML file
tree = ET.ElementTree(root)
with open("simulation.pvd", "wb") as f:
    tree.write(f, encoding="utf-8", xml_declaration=True)

print("simulation.pvd file created successfully.")
