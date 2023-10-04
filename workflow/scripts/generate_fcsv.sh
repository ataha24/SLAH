#!/bin/bash

# Expected arguments
SEGMENTATION=$1
OUTPUT_FCSV=$2

outputcent=$(c3d $SEGMENTATION -centroid | grep "CENTROID_MM")
coords=$(echo $outputcent | awk -F'[\\[\\]]' '{print $2}' | tr -d ' ')
IFS=',' read -ra ADDR <<< "$coords"
x=${ADDR[0]}
y=${ADDR[1]}
z=${ADDR[2]}
echo -e "# Markups fiducial file version = 4.10\n# CoordinateSystem = 0\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\nvtkMRMLMarkupsFiducialNode_1,$x,$y,$z,0.000,0.000,0.000,1.000,1,1,0,1,,vtkMRMLScalarVolumeNode1" > $OUTPUT_FCSV
