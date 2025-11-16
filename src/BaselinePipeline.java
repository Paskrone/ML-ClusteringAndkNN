import java.io.File;

/**
 * BaselinePipeline - Baseline-KNN ohne Clustering
 * 
 * Diese Pipeline implementiert den ersten Durchlauf:
 * - Lädt skalierten Diabetes-Datensatz
 * - Wendet direkt KNN an (OHNE Clustering)
 * - Evaluierung mit allen Metriken
 * - Hyperparameter-Tuning für k
 */
public class BaselinePipeline {
    
    /**
     * Führt Baseline-KNN auf skalierten Daten durch
     * 
     * @param trainFile Pfad zu skalierten Trainingsdaten
     * @param testFile Pfad zu skalierten Testdaten
     * @param kValues Array von k-Werten zum Testen
     * @return Bestes k und entsprechende Metriken
     */
    public static BaselineResult runBaseline(File trainFile, File testFile, int[] kValues) {
        System.out.println("=== BASELINE: KNN OHNE CLUSTERING ===\n");
        
        // 1. Daten laden
        System.out.println("1. Lade skalierten Diabetes-Datensatz...");
        double[][] trainData = Einlesen.einlesenDiabetes(trainFile);
        double[][] testData = Einlesen.einlesenDiabetes(testFile);
        
        // Features und Labels extrahieren
        // Achtung: skalierten Daten haben möglicherweise 9 oder 10 Spalten
        // Wir nehmen an, dass die letzte Spalte das Label ist
        double[][] trainFeatures = Einlesen.features(trainData);
        double[][] trainLabels = Einlesen.labelsBinary(trainData);
        double[][] testFeatures = Einlesen.features(testData);
        double[][] testLabels = Einlesen.labelsBinary(testData);
        
        System.out.println("   Trainingsdaten: " + trainFeatures.length + " Samples");
        System.out.println("   Testdaten: " + testFeatures.length + " Samples");
        System.out.println("   Features: " + trainFeatures[0].length + " Dimensionen\n");
        
        // 2. Hyperparameter-Tuning für k
        System.out.println("2. Hyperparameter-Tuning für k...");
        int bestK = kValues[0];
        double bestAccuracy = 0.0;
        
        for (int k : kValues) {
            // KNN-Modell trainieren
            KNearestNeighbor knn = new KNearestNeighbor(trainFeatures, trainLabels, k);
            
            // Vorhersagen für Testdaten
            int[] predictions = new int[testFeatures.length];
            int[] actual = new int[testLabels.length];
            for (int i = 0; i < testFeatures.length; i++) {
                predictions[i] = knn.predictClass(testFeatures[i]);
                actual[i] = (int) testLabels[i][0];
            }
            
            // Evaluation
            EvaluationMetrics metrics = new EvaluationMetrics(predictions, actual);
            double accuracy = metrics.getAccuracy();
            
            System.out.printf("   k=%2d: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f%n",
                    k, accuracy, metrics.getPrecision(), metrics.getRecall(), metrics.getF1Score());
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestK = k;
            }
        }
        
        // 3. Finale Ergebnisse mit bestem k
        System.out.println("\n3. Finale Evaluation mit bestem k=" + bestK + "...");
        KNearestNeighbor finalKNN = new KNearestNeighbor(trainFeatures, trainLabels, bestK);
        int[] finalPredictions = new int[testFeatures.length];
        int[] finalActual = new int[testLabels.length];
        for (int i = 0; i < testFeatures.length; i++) {
            finalPredictions[i] = finalKNN.predictClass(testFeatures[i]);
            finalActual[i] = (int) testLabels[i][0];
        }
        
        EvaluationMetrics finalMetrics = new EvaluationMetrics(finalPredictions, finalActual);
        System.out.println("\n=== BASELINE ERGEBNISSE (k=" + bestK + ") ===");
        finalMetrics.printAllMetrics();
        
        return new BaselineResult(bestK, finalMetrics);
    }
    
    /**
     * Ergebnis-Klasse für Baseline
     */
    public static class BaselineResult {
        public final int k;
        public final EvaluationMetrics metrics;
        
        public BaselineResult(int k, EvaluationMetrics metrics) {
            this.k = k;
            this.metrics = metrics;
        }
    }
}

