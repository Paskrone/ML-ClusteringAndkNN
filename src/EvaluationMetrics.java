/**
 * EvaluationMetrics - Berechnet verschiedene Metriken für binäre Klassifikation
 * 
 * Unterstützt:
 * - Accuracy
 * - Precision
 * - Recall
 * - F1-Score
 * - Confusion Matrix
 */
public class EvaluationMetrics {
    private int truePositives;   // TP: Vorhergesagt 1, Tatsächlich 1
    private int trueNegatives;   // TN: Vorhergesagt 0, Tatsächlich 0
    private int falsePositives;  // FP: Vorhergesagt 1, Tatsächlich 0
    private int falseNegatives;  // FN: Vorhergesagt 0, Tatsächlich 1
    
    /**
     * Konstruktor - berechnet Confusion Matrix aus Vorhersagen und tatsächlichen Labels
     * 
     * @param predictions Array mit vorhergesagten Klassen (0 oder 1)
     * @param actual Array mit tatsächlichen Klassen (0 oder 1)
     */
    public EvaluationMetrics(int[] predictions, int[] actual) {
        if (predictions.length != actual.length) {
            throw new IllegalArgumentException("Predictions und Actual müssen gleiche Länge haben!");
        }
        
        truePositives = 0;
        trueNegatives = 0;
        falsePositives = 0;
        falseNegatives = 0;
        
        for (int i = 0; i < predictions.length; i++) {
            int pred = predictions[i];
            int act = actual[i];
            
            if (pred == 1 && act == 1) {
                truePositives++;
            } else if (pred == 0 && act == 0) {
                trueNegatives++;
            } else if (pred == 1 && act == 0) {
                falsePositives++;
            } else if (pred == 0 && act == 1) {
                falseNegatives++;
            }
        }
    }
    
    /**
     * Berechnet Accuracy: (TP + TN) / (TP + TN + FP + FN)
     */
    public double getAccuracy() {
        int total = truePositives + trueNegatives + falsePositives + falseNegatives;
        if (total == 0) return 0.0;
        return (double)(truePositives + trueNegatives) / total;
    }
    
    /**
     * Berechnet Precision: TP / (TP + FP)
     * Maß für die Genauigkeit der positiven Vorhersagen
     */
    public double getPrecision() {
        int denominator = truePositives + falsePositives;
        if (denominator == 0) return 0.0;
        return (double)truePositives / denominator;
    }
    
    /**
     * Berechnet Recall (Sensitivity): TP / (TP + FN)
     * Maß dafür, wie viele positive Fälle korrekt erkannt wurden
     */
    public double getRecall() {
        int denominator = truePositives + falseNegatives;
        if (denominator == 0) return 0.0;
        return (double)truePositives / denominator;
    }
    
    /**
     * Berechnet F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
     * Harmonisches Mittel aus Precision und Recall
     */
    public double getF1Score() {
        double precision = getPrecision();
        double recall = getRecall();
        if (precision + recall == 0) return 0.0;
        return 2.0 * (precision * recall) / (precision + recall);
    }
    
    /**
     * Gibt die Confusion Matrix als String zurück
     */
    public void printConfusionMatrix() {
        System.out.println("\n=== Confusion Matrix ===");
        System.out.println("                    Tatsächlich");
        System.out.println("                Klasse 0    Klasse 1");
        System.out.println("Vorhergesagt 0    " + String.format("%5d", trueNegatives) + "      " + String.format("%5d", falseNegatives));
        System.out.println("Vorhergesagt 1    " + String.format("%5d", falsePositives) + "      " + String.format("%5d", truePositives));
        System.out.println();
    }
    
    /**
     * Gibt alle Metriken aus
     */
    public void printAllMetrics() {
        printConfusionMatrix();
        System.out.println("Accuracy:  " + String.format("%.4f", getAccuracy()));
        System.out.println("Precision: " + String.format("%.4f", getPrecision()));
        System.out.println("Recall:    " + String.format("%.4f", getRecall()));
        System.out.println("F1-Score:  " + String.format("%.4f", getF1Score()));
    }
    
    // Getter für direkten Zugriff
    public int getTruePositives() { return truePositives; }
    public int getTrueNegatives() { return trueNegatives; }
    public int getFalsePositives() { return falsePositives; }
    public int getFalseNegatives() { return falseNegatives; }
}


