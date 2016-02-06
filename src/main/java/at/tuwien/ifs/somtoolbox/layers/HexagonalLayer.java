package at.tuwien.ifs.somtoolbox.layers;
/*
 * Copyright 2004-2010 Information & Software Engineering Group (188/1)
 *                     Institute of Software Technology and Interactive Systems
 *                     Vienna University of Technology, Austria
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.ifs.tuwien.ac.at/dm/somtoolbox/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.awt.Point;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.ArrayUtils;

import cern.colt.matrix.DoubleFactory3D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.DoubleMatrix3D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.jet.math.Functions;

import at.tuwien.ifs.commons.util.MathUtils;
import at.tuwien.ifs.somtoolbox.SOMToolboxException;
import at.tuwien.ifs.somtoolbox.data.InputData;
import at.tuwien.ifs.somtoolbox.data.InputDatum;
import at.tuwien.ifs.somtoolbox.data.SOMLibClassInformation;
import at.tuwien.ifs.somtoolbox.data.SOMLibTemplateVector;
import at.tuwien.ifs.somtoolbox.input.InputCorrections;
import at.tuwien.ifs.somtoolbox.input.InputCorrections.InputCorrection;
import at.tuwien.ifs.somtoolbox.input.SOMLibFileFormatException;
import at.tuwien.ifs.somtoolbox.layers.AdaptiveCoordinatesVirtualLayer;
import at.tuwien.ifs.somtoolbox.layers.GrowingLayer;
import at.tuwien.ifs.somtoolbox.layers.InputContainer;
import at.tuwien.ifs.somtoolbox.layers.Layer;
import at.tuwien.ifs.somtoolbox.layers.Layer.GridLayout;
import at.tuwien.ifs.somtoolbox.layers.Layer.GridTopology;
import at.tuwien.ifs.somtoolbox.layers.LayerAccessException;
import at.tuwien.ifs.somtoolbox.layers.TrainingInterruptionListener;
import at.tuwien.ifs.somtoolbox.layers.Unit;
import at.tuwien.ifs.somtoolbox.layers.Unit.FeatureWeightMode;
import at.tuwien.ifs.somtoolbox.layers.metrics.AbstractMetric;
import at.tuwien.ifs.somtoolbox.layers.metrics.AbstractWeightedMetric;
import at.tuwien.ifs.somtoolbox.layers.metrics.DistanceMetric;
import at.tuwien.ifs.somtoolbox.layers.metrics.L2MetricSparse;
import at.tuwien.ifs.somtoolbox.layers.metrics.MetricException;
import at.tuwien.ifs.somtoolbox.layers.quality.AbstractQualityMeasure;
import at.tuwien.ifs.somtoolbox.layers.quality.QualityMeasure;
import at.tuwien.ifs.somtoolbox.layers.quality.QualityMeasureNotFoundException;
import at.tuwien.ifs.somtoolbox.models.GrowingSOM;
import at.tuwien.ifs.somtoolbox.properties.SOMProperties;
import at.tuwien.ifs.somtoolbox.properties.SOMProperties.SelectedClassMode;
import at.tuwien.ifs.somtoolbox.structures.ComponentLine2D;
import at.tuwien.ifs.somtoolbox.util.Cuboid;
import at.tuwien.ifs.somtoolbox.util.PCA;
import at.tuwien.ifs.somtoolbox.util.ProgressListener;
import at.tuwien.ifs.somtoolbox.util.ProgressListenerFactory;
import at.tuwien.ifs.somtoolbox.util.StdErrProgressWriter;
import at.tuwien.ifs.somtoolbox.util.StringUtils;
import at.tuwien.ifs.somtoolbox.util.VectorTools;
import at.tuwien.ifs.somtoolbox.util.comparables.ComponentRegionCount;
import at.tuwien.ifs.somtoolbox.util.comparables.InputDistance;
import at.tuwien.ifs.somtoolbox.util.comparables.InputNameDistance;
import at.tuwien.ifs.somtoolbox.util.comparables.UnitDistance;

/**
 * Implementation of a growing Self-Organizing Map layer that can also be static in size. Layer growth is based on the
 * quantization errors of the units and the distance to their respective neighboring units.
 * 
 * @author Michael Dittenbach
 * @author Rudolf Mayer
 * @version $Id: GrowingLayer.java 4300 2014-01-09 10:03:43Z mayer $
 */

public class HexagonalLayer extends GrowingLayer {
    

    public void initHex(){
    	gridLayout = GridLayout.hexagonal;
    }


    /**
     * Convenience constructor for top layer map of GHSOM or a single map. The identifier of the map is set to 1 and the
     * superordinate unit is set to <code>null</code>.
     * 
     * @param xSize the number of columns.
     * @param ySize the number of rows.
     * @param metricName the name of the distance metric to use.
     * @param dim the dimensionality of the weight vectors.
     * @param normalized the type of normalization that is applied to the weight vectors of newly created units. This is
     *            usually <code>Normalization.NONE</code> or <code>Normalization.UNIT_LEN</code>.
     * @param seed the random seed for creation of the units' weight vectors.
     * @see <a href="../data/normalisation/package.html">Normalisation</a>
     */
    public HexagonalLayer(int xSize, int ySize, String metricName, int dim, boolean normalized, boolean usePCA,
            long seed, InputData data) {
        this(xSize, ySize, 1, metricName, dim, normalized, usePCA, seed, data);
        initHex();
    }

    /**
     * Convenience constructor for top layer map of GHSOM or a single map. The identifier of the map is set to 1 and the
     * superordinate unit is set to <code>null</code>.
     * 
     * @param xSize the number of columns.
     * @param ySize the number of rows.
     * @param zSize the depth
     * @param metricName the name of the distance metric to use.
     * @param dim the dimensionality of the weight vectors.
     * @param normalized the type of normalization that is applied to the weight vectors of newly created units. This is
     *            usually <code>Normalization.NONE</code> or <code>Normalization.UNIT_LEN</code>.
     * @param seed the random seed for creation of the units' weight vectors.
     * @see <a href="../data/normalisation/package.html">Normalisation</a>
     */
    public HexagonalLayer(int xSize, int ySize, int zSize, String metricName, int dim, boolean normalized,
            boolean usePCA, long seed, InputData data) {
        this(1, null, xSize, ySize, zSize, metricName, dim, normalized, usePCA, seed, data);
        initHex();
    }

    /**
     * Constructor for a new, untrained layer.
     * 
     * @param id the unique id of the layer in a hierarchy.
     * @param su the pointer to the corresponding unit in the upper layer map.
     * @param xSize the number of units in horizontal direction.
     * @param ySize the number of units in vertical direction.
     * @param metricName the name of the distance metric to use.
     * @param dim the dimensionality of the weight vectors.
     * @param normalized the type of normalization that is applied to the weight vectors of newly created units. This is
     *            usually <code>Normalization.NONE</code> or <code>Normalization.UNIT_LEN</code>.
     * @param seed the random seed for creation of the units' weight vectors.
     * @see <a href="../data/normalisation/package.html">Normalisation</a>
     */
    public HexagonalLayer(int id, Unit su, int xSize, int ySize, String metricName, int dim, boolean normalized,
            boolean usePCA, long seed, InputData data) {
        this(id, su, xSize, ySize, 1, metricName, dim, normalized, usePCA, seed, data);
        initHex();
    }

    /**
     * Constructor for a new, untrained layer.
     * 
     * @param id the unique id of the layer in a hierarchy.
     * @param su the pointer to the corresponding unit in the upper layer map.
     * @param xSize the number of units in horizontal direction.
     * @param ySize the number of units in vertical direction.
     * @param zSize the number of units in depth
     * @param metricName the name of the distance metric to use.
     * @param dim the dimensionality of the weight vectors.
     * @param normalized the type of normalization that is applied to the weight vectors of newly created units. This is
     *            usually <code>Normalization.NONE</code> or <code>Normalization.UNIT_LEN</code>.
     * @param seed the random seed for creation of the units' weight vectors.
     * @see <a href="../data/normalisation/package.html">Normalisation</a>
     */
    public HexagonalLayer(int id, Unit su, int xSize, int ySize, int zSize, String metricName, int dim,
            boolean normalized, boolean usePCA, long seed, InputData data) {
    	
    	super(id, su, xSize, ySize, zSize, metricName, dim, normalized, usePCA, seed, data);
    	initHex();
    	
            }

    /**
     * Constructor for an already trained layer as specified by 2-dimensional array of d-dimensional weight vectors as
     * argument <code>vectors</code>.
     * 
     * @param xSize the number of columns.
     * @param ySize the number of rows.
     * @param metricName the name of the distance metric to use.
     * @param dim the dimensionality of the weight vectors.
     * @param vectors the three dimensional array of <code>d</code> dimensional weight vectors.
     * @param seed the random seed for creation of the units' weight vectors.
     * @throws SOMToolboxException if arguments <code>x</code>, <code>y</code> and <code>d</code> do not correspond to
     *             the dimensions of argument <code>vectors</code>.
     */
    public HexagonalLayer(int id, Unit su, int xSize, int ySize, String metricName, int dim, double[][][] vectors,
            long seed) throws SOMToolboxException {
        this(id, su, xSize, ySize, 1, metricName, dim, addDimension(xSize, ySize, vectors), seed);
        initHex();
    }

    

    /**
     * Constructor for an already trained layer as specified by 2-dimensional array of d-dimensional weight vectors as
     * argument <code>vectors</code>.
     * 
     * @param xSize the number of columns.
     * @param ySize the number of rows.
     * @param zSize the depth
     * @param metricName the name of the distance metric to use.
     * @param dim the dimensionality of the weight vectors.
     * @param vectors the two dimensional array of <code>d</code> dimensional weight vectors.
     * @param seed the random seed for creation of the units' weight vectors.
     * @throws SOMToolboxException if arguments <code>x</code>, <code>y</code> and <code>d</code> do not correspond to
     *             the dimensions of argument <code>vectors</code>.
     */
    public HexagonalLayer(int id, Unit su, int xSize, int ySize, int zSize, String metricName, int dim,
            double[][][][] vectors, long seed) throws SOMToolboxException {
    	super(id, su, xSize, ySize, zSize, metricName, dim, vectors, seed);
    	initHex();
        
    }



    public double getMapDistance(int x1, int y1, int x2, int y2) {
        return getMapDistance(x1, y1, 0, x2, y2, 0);
    }

    @Override
    public double getMapDistance(int x1, int y1, int z1, int x2, int y2, int z2) {
        return Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    }

    @Override
    public double getMapDistance(Unit u1, Unit u2) {
        return getMapDistance(u1.getXPos(), u1.getYPos(), u1.getZPos(), u2.getXPos(), u2.getYPos(), u2.getZPos());
    }

    public double getMapDistanceSq(int x1, int y1, int z1, int x2, int y2, int z2) {
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
    }

    public double getMapDistanceSq(Unit u1, Unit u2) {
        return getMapDistanceSq(u1.getXPos(), u1.getYPos(), u1.getZPos(), u2.getXPos(), u2.getYPos(), u2.getZPos());
    }

    
    public boolean hasNeighbours(int x, int y) throws LayerAccessException {
        if (x > 0 && getUnit(x - 1, y, 0) != null) {
            return true;
        } else if (x + 1 < units.length && getUnit(x + 1, y, 0) != null) {
            return true;
        } else if(y%2 == 1){
        	if (y > 0 && getUnit(x-1, y - 1, 0) != null) {
                return true;
            } else if (y > 0 && getUnit(x, y - 1, 0) != null) {
                return true;
            } else if (y + 1 < units[x-1].length && getUnit(x-1, y + 1, 0) != null) {
                return true;
            } else if (y + 1 < units[x].length && getUnit(x, y + 1, 0) != null) {
                return true;
            }
        } else if (y%2 == 0){
        	if (y > 0 && getUnit(x, y - 1, 0) != null) {
                return true;
            } else if (y > 0 && getUnit(x+1, y - 1, 0) != null) {
                return true;
            } else if (y + 1 < units[x].length && getUnit(x, y + 1, 0) != null) {
                return true;
            } else if (y + 1 < units[x+1].length && getUnit(x+1, y + 1, 0) != null) {
                return true;
            }
        }
        
        return false;
    }

    //1,3,5... x-1, x mod=1
    //2,4,6 ... x, x + 1, mod=0

    /**
     * Get direct neighbours of the given unit. Direct neighbours are neighbours in the same column or row of the SOM,
     * thus this method returns at most six neighbours (two for each of the x, y and z dimensions).
     */
    protected ArrayList<Unit> getNeighbouringUnits(Unit u) throws LayerAccessException {
        return getNeighbouringUnits(u.getXPos(), u.getYPos(), u.getZPos());
    }

    public ArrayList<Unit> getNeighbouringUnits(int x, int y) throws LayerAccessException {
        return getNeighbouringUnits(x, y, 0);
    }

    private ArrayList<Unit> getNeighbouringUnits(int x, int y, int z) throws LayerAccessException {
        ArrayList<Unit> neighbourUnits = new ArrayList<Unit>();

        if (x > 0) {
            neighbourUnits.add(getUnit(x - 1, y, z));
        }
        if (x + 1 < getXSize()) {
            neighbourUnits.add(getUnit(x + 1, y, z));
        }
        
      //1,3,5... x-1, x mod=1
        //2,4,6 ... x, x + 1, mod=0
        
        if(y%2 == 1){
	        if (y > 0) {
	            neighbourUnits.add(getUnit(x-1, y - 1, z));
	            neighbourUnits.add(getUnit(x, y - 1, z));
	        }
	        if (y + 1 < getYSize()) {
	            neighbourUnits.add(getUnit(x-1, y + 1, z));
	            neighbourUnits.add(getUnit(x, y + 1, z));

	        }
        } else if (y%2 == 0) {
        	if (y > 0) {
	            neighbourUnits.add(getUnit(x, y - 1, z));
	            neighbourUnits.add(getUnit(x+1, y - 1, z));

	        }
	        if (y + 1 < getYSize()) {
	            neighbourUnits.add(getUnit(x, y + 1, z));
	            neighbourUnits.add(getUnit(x+1, y + 1, z));

	        }
        }
        if (z > 0) {
            neighbourUnits.add(getUnit(x, y, z - 1));
        }
        if (z + 1 < getZSize()) {
            neighbourUnits.add(getUnit(x, y, z + 1));
        }
        return neighbourUnits;
    }


   
}
