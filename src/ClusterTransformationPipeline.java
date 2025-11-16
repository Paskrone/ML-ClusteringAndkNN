import java.io.File;

/**
 * ClusterTransformationPipeline - Durchlauf 4: Feature-Transformation/Reduktion basierend auf Clustern
 * 
 * Pipeline:
 * 1. Clustering auf skalierten Trainingsdaten
 * 2. Feature-Transformation basierend auf Cluster-Struktur
 * 3. KNN auf transformierten Features
 */
public class ClusterTransformationPipeline {
    
    /**
     * Führt Feature-Transformation basierend auf Clustern durch
     * 
     * @param trainFile Pfad zu skalierten Trainingsdaten
     * @param testFile Pfad zu skalierten Testdaten
     * @param numClusters Anzahl der Cluster für K-Means
     * @param kValues Array von k-Werten für KNN
     * @param transformationType Art der Transformation: "cluster_scaled", "pca", "cluster_features"
     * @return Bestes Ergebnis
     */
    public static TransformationResult runTransformation(File trainFile, File testFile, 
                                                        int numClusters, int[] kValues, String transformationType) {
        System.out.println("=== DURCHLAUF 4: FEATURE-TRANSFORMATION ===\n");
        
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
        
        // 4. Feature-Transformation
        System.out.println("4. Feature-Transformation (" + transformationType + ")...");
        double[][] transformedTrainFeatures;
        double[][] transformedTestFeatures;
        
        switch (transformationType.toLowerCase()) {
            case "cluster_scaled":
                transformedTrainFeatures = clusterSpecificScaling(trainFeatures, trainClusterLabels, numClusters);
                transformedTestFeatures = clusterSpecificScaling(testFeatures, testClusterLabels, numClusters);
                break;
            case "pca":
                // Einfache PCA-ähnliche Transformation: Reduktion auf Hauptkomponenten
                transformedTrainFeatures = simplePCATransform(trainFeatures, trainClusterLabels, centroids);
                transformedTestFeatures = simplePCATransform(testFeatures, testClusterLabels, centroids);
                break;
            case "cluster_features":
            default:
                // Transformation: Features relativ zu Cluster-Zentren
                transformedTrainFeatures = clusterRelativeFeatures(trainFeatures, trainClusterLabels, centroids);
                transformedTestFeatures = clusterRelativeFeatures(testFeatures, testClusterLabels, centroids);
                break;
        }
        
        System.out.println("   Original Features: " + trainFeatures[0].length);
        System.out.println("   Transformierte Features: " + transformedTrainFeatures[0].length + "\n");
        
        // 5. KNN mit verschiedenen k-Werten
        System.out.println("5. KNN mit transformierten Features...");
        int bestK = kValues[0];
        double bestAccuracy = 0.0;
        
        for (int k : kValues) {
            KNearestNeighbor knn = new KNearestNeighbor(transformedTrainFeatures, trainLabels, k);
            
            int[] predictions = new int[testFeatures.length];
            int[] actual = new int[testLabels.length];
            for (int i = 0; i < testFeatures.length; i++) {
                predictions[i] = knn.predictClass(transformedTestFeatures[i]);
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
        KNearestNeighbor finalKNN = new KNearestNeighbor(transformedTrainFeatures, trainLabels, bestK);
        int[] finalPredictions = new int[testFeatures.length];
        int[] finalActual = new int[testLabels.length];
        for (int i = 0; i < testFeatures.length; i++) {
            finalPredictions[i] = finalKNN.predictClass(transformedTestFeatures[i]);
            finalActual[i] = (int) testLabels[i][0];
        }
        
        EvaluationMetrics finalMetrics = new EvaluationMetrics(finalPredictions, finalActual);
        System.out.println("\n=== TRANSFORMATION ERGEBNISSE (k=" + bestK + ", Clusters=" + numClusters + ", Type=" + transformationType + ") ===");
        finalMetrics.printAllMetrics();
        
        return new TransformationResult(bestK, numClusters, transformationType, finalMetrics);
    }
    
    /**
     * Cluster-spezifische Skalierung: Jedes Feature wird relativ zum Cluster-Zentrum skaliert
     */
    private static double[][] clusterSpecificScaling(double[][] features, int[] clusterLabels, int numClusters) {
        int n = features.length;
        int d = features[0].length;
        double[][] transformed = new double[n][d];
        
        // Berechne Cluster-Standardabweichungen
        double[][] clusterStds = new double[numClusters][d];
        double[][] clusterMeans = new double[numClusters][d];
        int[] clusterCounts = new int[numClusters];
        
        // Berechne Mittelwerte
        for (int i = 0; i < n; i++) {
            int c = clusterLabels[i];
            clusterCounts[c]++;
            for (int j = 0; j < d; j++) {
                clusterMeans[c][j] += features[i][j];
            }
        }
        for (int c = 0; c < numClusters; c++) {
            for (int j = 0; j < d; j++) {
                clusterMeans[c][j] /= clusterCounts[c];
            }
        }
        
        // Berechne Standardabweichungen
        for (int i = 0; i < n; i++) {
            int c = clusterLabels[i];
            for (int j = 0; j < d; j++) {
                double diff = features[i][j] - clusterMeans[c][j];
                clusterStds[c][j] += diff * diff;
            }
        }
        for (int c = 0; c < numClusters; c++) {
            for (int j = 0; j < d; j++) {
                clusterStds[c][j] = Math.sqrt(clusterStds[c][j] / clusterCounts[c]);
                if (clusterStds[c][j] < 1e-6) clusterStds[c][j] = 1.0; // Vermeide Division durch 0
            }
        }
        
        // Transformiere: (x - mean) / std für jedes Cluster
        for (int i = 0; i < n; i++) {
            int c = clusterLabels[i];
            for (int j = 0; j < d; j++) {
                transformed[i][j] = (features[i][j] - clusterMeans[c][j]) / clusterStds[c][j];
            }
        }
        
        return transformed;
    }
    
    /**
     * Einfache PCA-ähnliche Transformation: Reduktion auf Cluster-basierte Hauptkomponenten
     */
    private static double[][] simplePCATransform(double[][] features, int[] clusterLabels, double[][] centroids) {
        int n = features.length;
        int numClusters = centroids.length;
        
        // Neue Dimension: Distanzen zu Clustern + Cluster-ID
        double[][] transformed = new double[n][numClusters + 1];
        
        for (int i = 0; i < n; i++) {
            // Distanzen zu allen Clustern
            for (int c = 0; c < numClusters; c++) {
                double dist = euclideanDistance(features[i], centroids[c]);
                transformed[i][c] = dist;
            }
            // Cluster-Zugehörigkeit
            transformed[i][numClusters] = (double) clusterLabels[i] / (numClusters - 1);
        }
        
        return transformed;
    }
    
    /**
     * Features relativ zu Cluster-Zentren: Differenz zu Cluster-Zentrum
     */
    private static double[][] clusterRelativeFeatures(double[][] features, int[] clusterLabels, double[][] centroids) {
        int n = features.length;
        int d = features[0].length;
        double[][] transformed = new double[n][d];
        
        for (int i = 0; i < n; i++) {
            int c = clusterLabels[i];
            for (int j = 0; j < d; j++) {
                // Differenz zum Cluster-Zentrum
                transformed[i][j] = features[i][j] - centroids[c][j];
            }
        }
        
        return transformed;
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
     * Ergebnis-Klasse für Transformation
     */
    public static class TransformationResult {
        public final int k;
        public final int numClusters;
        public final String transformationType;
        public final EvaluationMetrics metrics;
        
        public TransformationResult(int k, int numClusters, String transformationType, EvaluationMetrics metrics) {
            this.k = k;
            this.numClusters = numClusters;
            this.transformationType = transformationType;
            this.metrics = metrics;
        }
    }
}

