import java.io.File;

/**
 * ClusterFeatureExtensionPipeline - Durchlauf 3: Feature-Erweiterung mit Cluster-Informationen
 * 
 * Pipeline:
 * 1. Clustering auf skalierten Trainingsdaten
 * 2. Cluster-ID als zusätzliches Feature hinzufügen
 * 3. Optional: Distanz zu Cluster-Zentren als Features
 * 4. KNN auf erweiterten Features
 */
public class ClusterFeatureExtensionPipeline {
    
    /**
     * Führt Feature-Erweiterung mit Cluster-Informationen durch
     * 
     * @param trainFile Pfad zu skalierten Trainingsdaten
     * @param testFile Pfad zu skalierten Testdaten
     * @param numClusters Anzahl der Cluster für K-Means
     * @param kValues Array von k-Werten für KNN
     * @param includeDistances Ob Distanzen zu Clustern als Features hinzugefügt werden sollen
     * @return Bestes Ergebnis
     */
    public static FeatureExtensionResult runFeatureExtension(File trainFile, File testFile, 
                                                           int numClusters, int[] kValues, boolean includeDistances) {
        System.out.println("=== DURCHLAUF 3: FEATURE-ERWEITERUNG ===\n");
        
        // 1. Daten laden
        System.out.println("1. Lade skalierten Diabetes-Datensatz...");
        double[][] trainData = Einlesen.einlesenDiabetes(trainFile);
        double[][] testData = Einlesen.einlesenDiabetes(testFile);
        
        double[][] trainFeatures = Einlesen.features(trainData);
        double[][] trainLabels = Einlesen.labelsBinary(trainData);
        double[][] testFeatures = Einlesen.features(testData);
        double[][] testLabels = Einlesen.labelsBinary(testData);
        
        System.out.println("   Trainingsdaten: " + trainFeatures.length + " Samples");
        System.out.println("   Testdaten: " + testFeatures.length + " Samples\n");
        
        // 2. K-Means Clustering auf Trainingsdaten
        System.out.println("2. K-Means Clustering auf Trainingsdaten (k=" + numClusters + ")...");
        KMeansPascal kmeans = new KMeansPascal(numClusters, 300, "kmeans++", 42L);
        kmeans.fit(trainFeatures);
        int[] trainClusterLabels = kmeans.getLabels();
        double[][] centroids = kmeans.getCentroids();
        
        System.out.println("   Clustering abgeschlossen nach " + kmeans.getIterations() + " Iterationen\n");
        
        // 3. Cluster-Zuordnung für Testdaten
        System.out.println("3. Cluster-Zuordnung für Testdaten...");
        int[] testClusterLabels = kmeans.predictAll(testFeatures);
        
        // 4. Feature-Erweiterung
        System.out.println("4. Feature-Erweiterung...");
        double[][] extendedTrainFeatures = extendFeatures(trainFeatures, trainClusterLabels, centroids, includeDistances);
        double[][] extendedTestFeatures = extendFeatures(testFeatures, testClusterLabels, centroids, includeDistances);
        
        System.out.println("   Original Features: " + trainFeatures[0].length);
        System.out.println("   Erweiterte Features: " + extendedTrainFeatures[0].length);
        if (includeDistances) {
            System.out.println("   (Cluster-ID + " + numClusters + " Distanzen zu Clustern)\n");
        } else {
            System.out.println("   (Cluster-ID als zusätzliches Feature)\n");
        }
        
        // 5. KNN mit verschiedenen k-Werten
        System.out.println("5. KNN mit erweiterten Features...");
        int bestK = kValues[0];
        double bestAccuracy = 0.0;
        
        for (int k : kValues) {
            KNearestNeighbor knn = new KNearestNeighbor(extendedTrainFeatures, trainLabels, k);
            
            int[] predictions = new int[testFeatures.length];
            int[] actual = new int[testLabels.length];
            for (int i = 0; i < testFeatures.length; i++) {
                predictions[i] = knn.predictClass(extendedTestFeatures[i]);
                actual[i] = (int) testLabels[i][0];
            }
            
            EvaluationMetrics metrics = new EvaluationMetrics(predictions, actual);
            double accuracy = metrics.getAccuracy();
            
            System.out.printf("   k=%2d: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f%n",
                    k, accuracy, metrics.getPrecision(), metrics.getRecall(), metrics.getF1Score());
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestK = k;
            }
        }
        
        // 6. Finale Evaluation
        System.out.println("\n6. Finale Evaluation mit bestem k=" + bestK + "...");
        KNearestNeighbor finalKNN = new KNearestNeighbor(extendedTrainFeatures, trainLabels, bestK);
        int[] finalPredictions = new int[testFeatures.length];
        int[] finalActual = new int[testLabels.length];
        for (int i = 0; i < testFeatures.length; i++) {
            finalPredictions[i] = finalKNN.predictClass(extendedTestFeatures[i]);
            finalActual[i] = (int) testLabels[i][0];
        }
        
        EvaluationMetrics finalMetrics = new EvaluationMetrics(finalPredictions, finalActual);
        System.out.println("\n=== FEATURE-ERWEITERUNG ERGEBNISSE (k=" + bestK + ", Clusters=" + numClusters + ") ===");
        finalMetrics.printAllMetrics();
        
        return new FeatureExtensionResult(bestK, numClusters, finalMetrics);
    }
    
    /**
     * Erweitert Features um Cluster-Informationen
     */
    private static double[][] extendFeatures(double[][] features, int[] clusterLabels, 
                                             double[][] centroids, boolean includeDistances) {
        int n = features.length;
        int originalDim = features[0].length;
        int numClusters = centroids.length;
        
        // Neue Dimension: original + 1 (Cluster-ID) + optional numClusters (Distanzen)
        int newDim = originalDim + 1 + (includeDistances ? numClusters : 0);
        double[][] extended = new double[n][newDim];
        
        for (int i = 0; i < n; i++) {
            // Original Features kopieren
            System.arraycopy(features[i], 0, extended[i], 0, originalDim);
            
            // Cluster-ID als normiertes Feature hinzufügen (0-1 Skalierung)
            extended[i][originalDim] = (double) clusterLabels[i] / (numClusters - 1);
            
            // Optional: Distanzen zu allen Cluster-Zentren
            if (includeDistances) {
                for (int c = 0; c < numClusters; c++) {
                    double dist = euclideanDistance(features[i], centroids[c]);
                    extended[i][originalDim + 1 + c] = dist;
                }
            }
        }
        
        return extended;
    }
    
    /**
     * Berechnet euklidische Distanz zwischen zwei Vektoren
     */
    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Ergebnis-Klasse für Feature-Erweiterung
     */
    public static class FeatureExtensionResult {
        public final int k;
        public final int numClusters;
        public final EvaluationMetrics metrics;
        
        public FeatureExtensionResult(int k, int numClusters, EvaluationMetrics metrics) {
            this.k = k;
            this.numClusters = numClusters;
            this.metrics = metrics;
        }
    }
}

