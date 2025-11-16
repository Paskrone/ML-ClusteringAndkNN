import java.io.File;
import java.util.*;

/**
 * ClusterImputationPipeline - Durchlauf 2: Cluster-basierte Datenimputation
 * 
 * Pipeline:
 * 1. Clustering auf skalierten Trainingsdaten
 * 2. Identifikation fehlender Werte (0 bei bestimmten Features)
 * 3. Imputation durch Cluster-Median/Mean
 * 4. KNN auf imputierten Daten
 */
public class ClusterImputationPipeline {
    
    // Features, bei denen 0 als fehlender Wert interpretiert wird
    private static final int[] MISSING_VALUE_FEATURES = {1, 2, 3, 4, 5}; // Glucose, BloodPressure, SkinThickness, Insulin, BMI
    
    /**
     * Führt Cluster-basierte Imputation durch
     * 
     * @param trainFile Pfad zu skalierten Trainingsdaten
     * @param testFile Pfad zu skalierten Testdaten
     * @param numClusters Anzahl der Cluster für K-Means
     * @param kValues Array von k-Werten für KNN
     * @return Bestes Ergebnis
     */
    public static ImputationResult runImputation(File trainFile, File testFile, int numClusters, int[] kValues) {
        System.out.println("=== DURCHLAUF 2: CLUSTER-BASIERTE IMPUTATION ===\n");
        
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
        
        System.out.println("   Clustering abgeschlossen nach " + kmeans.getIterations() + " Iterationen");
        System.out.println("   Inertia (SSE): " + String.format("%.4f", kmeans.getInertia()) + "\n");
        
        // 3. Cluster-Zuordnung für Testdaten
        System.out.println("3. Cluster-Zuordnung für Testdaten...");
        int[] testClusterLabels = kmeans.predictAll(testFeatures);
        
        // 4. Cluster-basierte Imputation
        System.out.println("4. Cluster-basierte Imputation...");
        double[][] imputedTrainFeatures = imputeMissingValues(trainFeatures, trainClusterLabels, centroids, numClusters);
        double[][] imputedTestFeatures = imputeMissingValues(testFeatures, testClusterLabels, centroids, numClusters);
        
        // 5. KNN mit verschiedenen k-Werten
        System.out.println("5. KNN mit imputierten Daten...");
        int bestK = kValues[0];
        double bestAccuracy = 0.0;
        
        for (int k : kValues) {
            KNearestNeighbor knn = new KNearestNeighbor(imputedTrainFeatures, trainLabels, k);
            
            int[] predictions = new int[testFeatures.length];
            int[] actual = new int[testLabels.length];
            for (int i = 0; i < testFeatures.length; i++) {
                predictions[i] = knn.predictClass(imputedTestFeatures[i]);
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
        KNearestNeighbor finalKNN = new KNearestNeighbor(imputedTrainFeatures, trainLabels, bestK);
        int[] finalPredictions = new int[testFeatures.length];
        int[] finalActual = new int[testLabels.length];
        for (int i = 0; i < testFeatures.length; i++) {
            finalPredictions[i] = finalKNN.predictClass(imputedTestFeatures[i]);
            finalActual[i] = (int) testLabels[i][0];
        }
        
        EvaluationMetrics finalMetrics = new EvaluationMetrics(finalPredictions, finalActual);
        System.out.println("\n=== IMPUTATION ERGEBNISSE (k=" + bestK + ", Clusters=" + numClusters + ") ===");
        finalMetrics.printAllMetrics();
        
        return new ImputationResult(bestK, numClusters, finalMetrics);
    }
    
    /**
     * Führt Cluster-basierte Imputation durch
     * Ersetzt 0-Werte durch Cluster-Median
     */
    private static double[][] imputeMissingValues(double[][] features, int[] clusterLabels, 
                                                   double[][] centroids, int numClusters) {
        int n = features.length;
        int d = features[0].length;
        double[][] imputed = new double[n][d];
        
        // Kopiere alle Werte
        for (int i = 0; i < n; i++) {
            System.arraycopy(features[i], 0, imputed[i], 0, d);
        }
        
        // Berechne Cluster-Mediane für jedes Feature
        double[][] clusterMedians = new double[numClusters][d];
        for (int c = 0; c < numClusters; c++) {
            @SuppressWarnings("unchecked")
            List<Double>[] featureValues = new List[d];
            for (int j = 0; j < d; j++) {
                featureValues[j] = new ArrayList<>();
            }
            
            // Sammle Werte für dieses Cluster
            for (int i = 0; i < n; i++) {
                if (clusterLabels[i] == c) {
                    for (int j = 0; j < d; j++) {
                        featureValues[j].add(features[i][j]);
                    }
                }
            }
            
            // Berechne Median für jedes Feature
            for (int j = 0; j < d; j++) {
                if (featureValues[j].isEmpty()) {
                    clusterMedians[c][j] = centroids[c][j]; // Fallback auf Centroid
                } else {
                    Collections.sort(featureValues[j]);
                    int size = featureValues[j].size();
                    if (size % 2 == 0) {
                        clusterMedians[c][j] = (featureValues[j].get(size/2 - 1) + featureValues[j].get(size/2)) / 2.0;
                    } else {
                        clusterMedians[c][j] = featureValues[j].get(size/2);
                    }
                }
            }
        }
        
        // Imputiere fehlende Werte (0 bei bestimmten Features)
        int imputedCount = 0;
        for (int i = 0; i < n; i++) {
            int cluster = clusterLabels[i];
            for (int featureIdx : MISSING_VALUE_FEATURES) {
                if (featureIdx < d && Math.abs(imputed[i][featureIdx]) < 1e-6) { // 0 oder sehr nahe bei 0
                    imputed[i][featureIdx] = clusterMedians[cluster][featureIdx];
                    imputedCount++;
                }
            }
        }
        
        System.out.println("   " + imputedCount + " fehlende Werte imputiert");
        return imputed;
    }
    
    /**
     * Ergebnis-Klasse für Imputation
     */
    public static class ImputationResult {
        public final int k;
        public final int numClusters;
        public final EvaluationMetrics metrics;
        
        public ImputationResult(int k, int numClusters, EvaluationMetrics metrics) {
            this.k = k;
            this.numClusters = numClusters;
            this.metrics = metrics;
        }
    }
}

