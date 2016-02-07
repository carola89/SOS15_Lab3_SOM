package at.tuwien.ifs.somtoolbox.models;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.geom.Point2D;
import java.awt.geom.Point2D.Double;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import at.tuwien.ifs.somtoolbox.SOMToolboxException;
import at.tuwien.ifs.somtoolbox.layers.GrowingLayer;
import at.tuwien.ifs.somtoolbox.layers.HexagonalLayer;
import at.tuwien.ifs.somtoolbox.layers.Unit;
import at.tuwien.ifs.somtoolbox.util.ImageUtils;
import at.tuwien.ifs.somtoolbox.util.VisualisationUtils;
import at.tuwien.ifs.somtoolbox.visualization.FuzzyColourCodingVisualiser;
import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.math.Functions;

public class MyFuzzyColourCodingVisualiser extends FuzzyColourCodingVisualiser {
	
	@Override
    public BufferedImage createVisualization(int variantIndex, GrowingSOM gsom, int width, int height)
            throws SOMToolboxException {

        GrowingLayer layer = gsom.getLayer();

        BufferedImage res = ImageUtils.createEmptyImage(width, height);
        Graphics2D g = (Graphics2D) res.getGraphics();

        double unitWidth = width / (double) layer.getXSize();
        double unitHeight = height / (double) layer.getYSize();

        // set up array of unit coordinates
        Point2D.Double[][] locations = new Point2D.Double[layer.getXSize()][layer.getYSize()];
        for (int i = 0; i < layer.getXSize(); i++) {
            for (int j = 0; j < layer.getYSize(); j++) {
                locations[i][j] = new Double(i, j);
            }
        }

        // construct a dissimilarity matrix of the model vectors
        DoubleMatrix2D unitDistanceMatrix = layer.getUnitDistanceMatrix();

        // transform to a similarity matrix - Equation (1) in Himberg 2000.
        DoubleMatrix2D similarityMatrix = unitDistanceMatrix.copy();
        similarityMatrix.assign(new DoubleFunction() {
           // @Override
            public double apply(double argument) {
                return Math.exp(-(argument * argument / T));
            }
        });

        // normalise each row so that it sums up to 1
        for (int i = 0; i < similarityMatrix.rows(); i++) {
            DoubleMatrix1D row = similarityMatrix.viewRow(i);
            final double sum = row.aggregate(Functions.plus, Functions.identity);
            row.assign(new DoubleFunction() {
                //@Override
                public double apply(double argument) {
                    return argument / sum;
                }
            });
        }

        // contraction process
        // FIXME: check this with the Matlab implementation, it seems that is a bit different to the paper
        // http://www.cis.hut.fi/somtoolbox/package/docs2/som_fuzzycolor.html)
        for (int k = 0; k < r; k++) {
            Double[][] newLocations = new Double[layer.getXSize()][layer.getYSize()];
            for (int x = 0; x < layer.getXSize(); x++) {
                for (int y = 0; y < layer.getYSize(); y++) {
                    Double loc = locations[x][y];
                    Double newLoc = new Double(loc.x, loc.y);
                    int unitIndex = layer.getUnitIndex(x, y);
                    for (int x1 = 0; x1 < layer.getXSize(); x1++) {
                        for (int y1 = 0; y1 < layer.getYSize(); y1++) {
                            if (x != x1 && y != y1) {
                                int otherUnitIndex = layer.getUnitIndex(x1, y1);
                                double similarity = similarityMatrix.getQuick(unitIndex, otherUnitIndex);
                                // move towards that location
                                double diffX = locations[x1][y1].x - loc.x;
                                double diffY = locations[x1][y1].y - loc.y;
                                newLoc.setLocation(newLoc.x + diffX * similarity, newLoc.y + diffY * similarity);
                            }
                        }
                    }
                    newLocations[x][y] = newLoc;
                }
            }
            locations = newLocations;
        }

        // obtain RGB slice according to the (contracted) unit positions, and draw visualisation
        Color[][] colours = new Color[locations.length][locations[0].length];

        if (showColourCoding) {
            double colourZoomX = 255.0 / layer.getXSize();
            double colourZoomY = 255.0 / layer.getYSize();
            for (int i = 0; i < layer.getXSize(); i++) {
                for (int j = 0; j < layer.getYSize(); j++) {
                    Double loc = locations[i][j];

                    // colour the SOM unit
                    colours[i][j] = new Color(
                    // red is 255 on the top, and 0 on the bottom
                            (int) Math.round(colourZoomY * (layer.getYSize() - loc.y)),
                            // green is 255 on the left, and 0 on the right
                            (int) Math.round(colourZoomX * (layer.getXSize() - loc.x)),
                            // blue is 0 on the top, and 255 on the bottom
                            (int) Math.round(colourZoomY * loc.y));

                    g.setColor(colours[i][j]);
                    g.fillRect((int) (i * unitWidth), (int) (j * unitHeight), (int) unitWidth, (int) unitHeight);
                }
            }
        }

        if (showUnitNodes) {
            int markerHeight = (int) (unitHeight / 5);
            int markerWidth = (int) (unitWidth / 5);
            for (int i = 0; i < layer.getXSize(); i++) {
                for (int j = 0; j < layer.getYSize(); j++) {
                    Double loc = locations[i][j];
                    // draw the nodes
                    g.setColor(Color.black);
                    Point markerPos = getMarkerPos(unitWidth, unitHeight, markerWidth, markerHeight, loc);
                    VisualisationUtils.drawMarker(g, markerWidth, markerHeight, markerPos);
                }
            }
        }

        if (showConnectingLines) {
            g.setColor(Color.black);
            int lineWidth = (int) Math.round(unitWidth / 20);
            int lineHeight = (int) Math.round(unitHeight / 20);
            // draw the connections between nodes; can do this only after colouring, as it needs to be on top
            for (int i = 0; i < layer.getXSize(); i++) {
                for (int j = 0; j < layer.getYSize(); j++) {
                	
                	Point start = getLinePos(unitWidth, unitHeight, locations[i][j]);
                	// draw the nodes connections to the right
                    if (i + 1 < layer.getXSize()) {
                        Point end = getLinePos(unitWidth, unitHeight, locations[i + 1][j]);
                        VisualisationUtils.drawThickLine(g, start.x, start.y, end.x, end.y, lineWidth, lineHeight);
                    }
                	 //1,3,5... x-1, x mod=1
                    //2,4,6 ... x, x + 1, mod=0
                	if(j % 2 == 1){
                        // draw the nodes connections to the right
                        if (j + 1 < layer.getYSize()) {
                            Point end = getLinePos(unitWidth, unitHeight, locations[i][j + 1]);
                            VisualisationUtils.drawThickLine(g, start.x, start.y, end.x, end.y, lineWidth, lineHeight);
                            if (i-1 >= 0){
                            	Point end2 = getLinePos(unitWidth, unitHeight, locations[i-1][j + 1]);
                                VisualisationUtils.drawThickLine(g, start.x, start.y, end2.x, end2.y, lineWidth, lineHeight);
                            }
                        } 
                        
                	} else {
                		// draw the nodes connections to the right
                        if (j + 1 < layer.getYSize()) {
                            Point end = getLinePos(unitWidth, unitHeight, locations[i][j + 1]);
                            VisualisationUtils.drawThickLine(g, start.x, start.y, end.x, end.y, lineWidth, lineHeight);
                            if (i+1  < layer.getXSize()){
                            	Point end2 = getLinePos(unitWidth, unitHeight, locations[i+1][j + 1]);
                                VisualisationUtils.drawThickLine(g, start.x, start.y, end2.x, end2.y, lineWidth, lineHeight);
                            }
                       } 
                        
                	}
                    
                }
            }
        }

        return res;
    }
	
	private Point getMarkerPos(double unitWidth, double unitHeight, int markerWidth, int markerHeight, Double loc) {
        return new Point((int) Math.round(loc.x * unitWidth + (unitWidth - markerWidth) / 2), (int) Math.round(loc.y
                * unitHeight + (unitHeight - markerHeight) / 2));
    }
	
	 private Point getLinePos(double unitWidth, double unitHeight, Double loc) {
	        return new Point((int) Math.round(loc.x * unitWidth + unitWidth / 2), (int) Math.round(loc.y * unitHeight
	                + unitHeight / 2));
	    }
}
