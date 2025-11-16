# Fallstudie: Verbesserung der Vorhersagequalität durch Clustering

## Projektübersicht

Dieses Projekt implementiert eine Machine-Learning-Pipeline zur Untersuchung, ob Clustering-Methoden die Vorhersagequalität eines Klassifikationsmodells verbessern können. Als Datengrundlage dient der Diabetes-Datensatz (`diabetes.csv`).

**Zentrale Frage:** Wie kann Clustering zur Verbesserung der Prediction-Performance beitragen?

## Projektstruktur

```
ML/
├── src/                          # Quellcode (Java-Dateien)
│   ├── EvaluationMetrics.java    # Metriken-Berechnung
│   ├── BaselinePipeline.java     # Baseline ohne Clustering
│   ├── ClusterImputationPipeline.java      # Durchlauf 2: Imputation
│   ├── ClusterFeatureExtensionPipeline.java # Durchlauf 3: Feature-Erweiterung
│   ├── ClusterTransformationPipeline.java   # Durchlauf 4: Transformation
│   ├── HauptprogrammVL.java      # Hauptprogramm mit allen Methoden
│   └── ...                       # Weitere bestehende Klassen
├── bin/                          # Kompilierter Code (.class Dateien)
├── data/                         # Datensätze
│   ├── diabetes.csv              # Original-Datensatz
│   ├── diabetes_scaled_train.csv  # Skalierte Trainingsdaten
│   └── diabetes_scaled_test.csv  # Skalierte Testdaten
└── README.md                     # Diese Datei
```

## Implementierte Durchläufe

Das Projekt besteht aus **4 unabhängigen Durchläufen**, die verschiedene Ansätze zur Integration von Clustering testen:

### Durchlauf 1: Baseline (OHNE Clustering)
- **Datei:** `BaselinePipeline.java`
- **Ansatz:** Direkt KNN auf skalierten Daten, ohne Clustering
- **Ziel:** Referenz-Benchmark (~75% Accuracy)

### Durchlauf 2: Cluster-basierte Datenimputation
- **Datei:** `ClusterImputationPipeline.java`
- **Ansatz:** 
  1. K-Means Clustering auf Trainingsdaten
  2. Identifikation fehlender Werte (0 bei Glucose, BloodPressure, SkinThickness, Insulin, BMI)
  3. Imputation durch Cluster-Median
  4. KNN auf imputierten Daten

### Durchlauf 3: Feature-Erweiterung
- **Datei:** `ClusterFeatureExtensionPipeline.java`
- **Ansatz:**
  1. K-Means Clustering auf Trainingsdaten
  2. Cluster-ID als zusätzliches Feature
  3. Distanzen zu allen Cluster-Zentren als Features
  4. KNN auf erweiterten Features (8 → 13 Features)

### Durchlauf 4: Feature-Transformation
- **Datei:** `ClusterTransformationPipeline.java`
- **Ansatz:**
  1. K-Means Clustering auf Trainingsdaten
  2. Feature-Transformation: Features relativ zu Cluster-Zentren
  3. KNN auf transformierten Features

## Klassen-Dokumentation

### EvaluationMetrics.java

**Zweck:** Berechnet verschiedene Metriken für binäre Klassifikation.

**Hauptmethoden:**
- `EvaluationMetrics(int[] predictions, int[] actual)` - Konstruktor, berechnet Confusion Matrix
- `getAccuracy()` - (TP + TN) / (TP + TN + FP + FN)
- `getPrecision()` - TP / (TP + FP)
- `getRecall()` - TP / (TP + FN)
- `getF1Score()` - 2 * (Precision * Recall) / (Precision + Recall)
- `printConfusionMatrix()` - Gibt Confusion Matrix aus
- `printAllMetrics()` - Gibt alle Metriken aus

**Verwendung:**
```java
int[] predictions = {0, 1, 0, 1, ...};
int[] actual = {0, 1, 1, 1, ...};
EvaluationMetrics metrics = new EvaluationMetrics(predictions, actual);
System.out.println("Accuracy: " + metrics.getAccuracy());
```

### BaselinePipeline.java

**Zweck:** Implementiert Baseline-KNN ohne Clustering.

**Hauptmethode:**
- `runBaseline(File trainFile, File testFile, int[] kValues)` - Führt Baseline-Pipeline aus

**Ablauf:**
1. Lädt skalierten Diabetes-Datensatz
2. Hyperparameter-Tuning für k (testet verschiedene k-Werte)
3. Evaluation mit allen Metriken
4. Gibt bestes k und Ergebnisse zurück

**Verwendung:**
```java
File trainFile = new File("data/diabetes_scaled_train.csv");
File testFile = new File("data/diabetes_scaled_test.csv");
int[] kValues = {5, 10, 15, 20, 25, 30, 55};
BaselinePipeline.BaselineResult result = BaselinePipeline.runBaseline(trainFile, testFile, kValues);
```

### ClusterImputationPipeline.java

**Zweck:** Implementiert Cluster-basierte Datenimputation.

**Hauptmethode:**
- `runImputation(File trainFile, File testFile, int numClusters, int[] kValues)` - Führt Imputation-Pipeline aus

**Ablauf:**
1. Lädt skalierten Datensatz
2. K-Means Clustering auf Trainingsdaten
3. Cluster-Zuordnung für Testdaten
4. Identifikation fehlender Werte (0 bei bestimmten Features)
5. Imputation durch Cluster-Median
6. KNN auf imputierten Daten

**Wichtige Details:**
- Fehlende Werte werden bei folgenden Features erkannt: Glucose (Index 1), BloodPressure (2), SkinThickness (3), Insulin (4), BMI (5)
- Imputation verwendet den Median aller Werte im gleichen Cluster für das jeweilige Feature

**Verwendung:**
```java
int numClusters = 4;
ClusterImputationPipeline.ImputationResult result = 
    ClusterImputationPipeline.runImputation(trainFile, testFile, numClusters, kValues);
```

### ClusterFeatureExtensionPipeline.java

**Zweck:** Implementiert Feature-Erweiterung mit Cluster-Informationen.

**Hauptmethode:**
- `runFeatureExtension(File trainFile, File testFile, int numClusters, int[] kValues, boolean includeDistances)` - Führt Feature-Erweiterung aus

**Ablauf:**
1. K-Means Clustering auf Trainingsdaten
2. Cluster-Zuordnung für alle Datenpunkte
3. Feature-Erweiterung:
   - Cluster-ID als normiertes Feature (0-1 Skalierung)
   - Optional: Euklidische Distanzen zu allen Cluster-Zentren
4. KNN auf erweiterten Features

**Feature-Dimensionen:**
- Original: 8 Features
- Mit Cluster-ID: 9 Features
- Mit Cluster-ID + Distanzen: 8 + 1 + 4 = 13 Features

**Verwendung:**
```java
boolean includeDistances = true; // Distanzen als Features hinzufügen
ClusterFeatureExtensionPipeline.FeatureExtensionResult result = 
    ClusterFeatureExtensionPipeline.runFeatureExtension(trainFile, testFile, numClusters, kValues, includeDistances);
```

### ClusterTransformationPipeline.java

**Zweck:** Implementiert Feature-Transformation basierend auf Clustern.

**Hauptmethode:**
- `runTransformation(File trainFile, File testFile, int numClusters, int[] kValues, String transformationType)` - Führt Transformation aus

**Transformationstypen:**
- `"cluster_features"` (Standard): Features relativ zu Cluster-Zentren (x - centroid)
- `"cluster_scaled"`: Cluster-spezifische Standardisierung
- `"pca"`: PCA-ähnliche Transformation auf Cluster-Distanzen

**Ablauf:**
1. K-Means Clustering auf Trainingsdaten
2. Cluster-Zuordnung für alle Datenpunkte
3. Feature-Transformation je nach Typ
4. KNN auf transformierten Features

**Verwendung:**
```java
String transformationType = "cluster_features";
ClusterTransformationPipeline.TransformationResult result = 
    ClusterTransformationPipeline.runTransformation(trainFile, testFile, numClusters, kValues, transformationType);
```

### HauptprogrammVL.java

**Zweck:** Hauptprogramm mit allen Methoden zur Ausführung der Durchläufe.

**Wichtige Methoden:**
- `baselineKNN_Diabetes()` - Führt Durchlauf 1 aus
- `imputationKNN_Diabetes()` - Führt Durchlauf 2 aus
- `featureExtensionKNN_Diabetes()` - Führt Durchlauf 3 aus
- `transformationKNN_Diabetes()` - Führt Durchlauf 4 aus
- `compareAllApproaches()` - Führt alle 4 Durchläufe nacheinander aus und vergleicht sie

**Verwendung:**
```java
// Einzelne Durchläufe
HauptprogrammVL.baselineKNN_Diabetes();
HauptprogrammVL.imputationKNN_Diabetes();

// Alle Durchläufe auf einmal
HauptprogrammVL.compareAllApproaches();
```

## Kompilierung und Ausführung

### Voraussetzungen
- Java JDK (Version 8 oder höher)
- Skalierte Diabetes-Daten in `data/diabetes_scaled_train.csv` und `data/diabetes_scaled_test.csv`

### Kompilierung

```bash
# Alle neuen Dateien kompilieren
javac -encoding UTF-8 -d bin -cp bin src/EvaluationMetrics.java src/BaselinePipeline.java src/ClusterImputationPipeline.java src/ClusterFeatureExtensionPipeline.java src/ClusterTransformationPipeline.java src/HauptprogrammVL.java

# Oder alle Java-Dateien kompilieren
javac -encoding UTF-8 -d bin -cp bin src/*.java
```

### Ausführung

```bash
# Alle Durchläufe ausführen
java -cp bin HauptprogrammVL

# Oder einzelne Durchläufe in der main-Methode aktivieren
```

## Testergebnisse

### Durchlauf 1: Baseline (OHNE Clustering)
- **Accuracy:** 74.03%
- **Precision:** 0.6400
- **Recall:** 0.5926
- **F1-Score:** 0.6154
- **Bester k:** 25

### Durchlauf 2: Cluster-basierte Imputation
- **Accuracy:** 73.38% (-0.65%)
- **Precision:** 0.6444
- **Recall:** 0.5370
- **F1-Score:** 0.5859
- **Bester k:** 55
- **Imputierte Werte:** 511 (Train), 141 (Test)

### Durchlauf 3: Feature-Erweiterung
- **Accuracy:** 70.78% (-3.25%)
- **Precision:** 0.5714
- **Recall:** 0.6667
- **F1-Score:** 0.6154
- **Bester k:** 20
- **Erweiterte Features:** 8 → 13

### Durchlauf 4: Feature-Transformation
- **Accuracy:** 73.38% (-0.65%)
- **Precision:** 0.6857
- **Recall:** 0.4444
- **F1-Score:** 0.5393
- **Bester k:** 30
- **Transformation:** cluster_features

## Interpretation der Ergebnisse

### Haupterkenntnisse

1. **Baseline ist am besten:** Die Baseline-Methode ohne Clustering erreicht mit 74.03% die höchste Accuracy.

2. **Clustering verbessert nicht:** Keiner der Clustering-Ansätze konnte die Baseline übertreffen:
   - Imputation: -0.65%
   - Feature-Erweiterung: -3.25%
   - Transformation: -0.65%

3. **Mögliche Gründe:**
   - Die skalierten Daten sind bereits gut vorbereitet
   - Die Cluster-Struktur passt möglicherweise nicht optimal zur Klassifikationsaufgabe
   - Clustering fügt hier keine zusätzliche Information hinzu, die für die Klassifikation nützlich ist

4. **Feature-Erweiterung schneidet am schlechtesten ab:** Die Hinzufügung von Cluster-Informationen als Features reduziert die Performance deutlich (-3.25%).

## Technische Details

### K-Means Clustering
- **Implementierung:** `KMeansPascal.java`
- **Initialisierung:** K-Means++ (intelligente Startwerte)
- **Maximale Iterationen:** 300
- **Abbruchkriterium:** SSE (Sum of Squared Errors) stabilisiert sich
- **Standard Clusteranzahl:** 4

### K-Nearest Neighbors (KNN)
- **Implementierung:** `KNearestNeighbor.java`
- **Distanzmetrik:** Euklidische Distanz
- **Getestete k-Werte:** 5, 10, 15, 20, 25, 30, 55
- **Klassifikation:** Mehrheitsentscheidung der k nächsten Nachbarn

### Datenverarbeitung
- **Skalierung:** MinMaxScaler (Werte auf [0,1] normalisiert)
- **Train/Test Split:** 80/20 (bereits in skalierten Dateien)
- **Features:** 8 Dimensionen (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Label:** Binär (0 = kein Diabetes, 1 = Diabetes)

## Pipeline-Ablauf (Detailliert)

### Durchlauf 1: Baseline
```
1. Daten laden (diabetes_scaled_train.csv, diabetes_scaled_test.csv)
2. Features und Labels extrahieren
3. Für jedes k in kValues:
   a. KNN-Modell trainieren
   b. Vorhersagen für Testdaten
   c. Evaluation mit allen Metriken
4. Bestes k auswählen
5. Finale Evaluation mit bestem k
```

### Durchlauf 2: Imputation
```
1. Daten laden
2. K-Means Clustering auf Trainingsdaten
3. Cluster-Zuordnung für Testdaten
4. Für jeden Datenpunkt:
   a. Fehlende Werte identifizieren (0 bei bestimmten Features)
   b. Cluster-Median für fehlende Werte berechnen
   c. Fehlende Werte ersetzen
5. KNN auf imputierten Daten (wie Durchlauf 1)
```

### Durchlauf 3: Feature-Erweiterung
```
1. Daten laden
2. K-Means Clustering auf Trainingsdaten
3. Cluster-Zuordnung für alle Datenpunkte
4. Für jeden Datenpunkt:
   a. Cluster-ID als normiertes Feature hinzufügen
   b. Euklidische Distanzen zu allen Cluster-Zentren berechnen
   c. Distanzen als Features hinzufügen
5. KNN auf erweiterten Features (wie Durchlauf 1)
```

### Durchlauf 4: Transformation
```
1. Daten laden
2. K-Means Clustering auf Trainingsdaten
3. Cluster-Zuordnung für alle Datenpunkte
4. Für jeden Datenpunkt:
   a. Features relativ zu Cluster-Zentrum transformieren (x - centroid)
5. KNN auf transformierten Features (wie Durchlauf 1)
```

## Anpassungsmöglichkeiten

### Clusteranzahl ändern
In `HauptprogrammVL.java`:
```java
int numClusters = 4; // Ändern zu 3, 5, 6, etc.
```

### Andere k-Werte testen
In `HauptprogrammVL.java`:
```java
int[] kValues = {5, 10, 15, 20, 25, 30, 55}; // Anpassen
```

### Andere Transformationstypen
In `transformationKNN_Diabetes()`:
```java
String transformationType = "cluster_scaled"; // oder "pca"
```

### Distanzen in Feature-Erweiterung deaktivieren
In `featureExtensionKNN_Diabetes()`:
```java
boolean includeDistances = false; // Nur Cluster-ID, keine Distanzen
```

## Bekannte Einschränkungen

1. **Encoding-Probleme:** Einige bestehende Dateien haben UTF-8 Encoding-Probleme. Die neuen Dateien verwenden korrektes UTF-8.

2. **Fehlende Werte:** Die Imputation identifiziert nur 0-Werte bei bestimmten Features. Andere fehlende Werte werden nicht behandelt.

3. **Clusteranzahl:** Die optimale Clusteranzahl wird nicht automatisch bestimmt (z.B. durch Elbow-Methode). Aktuell fest auf 4 gesetzt.

4. **Skalierung:** Die Daten müssen bereits skaliert sein. Die Pipeline skaliert nicht selbst.

## Zukünftige Verbesserungen

- Automatische Bestimmung der optimalen Clusteranzahl (Elbow-Methode, Silhouette-Score)
- Cross-Validation für robustere Ergebnisse
- Weitere Transformationstypen (z.B. PCA, t-SNE)
- Visualisierung der Cluster-Struktur
- Vergleich mit anderen Clustering-Algorithmen (DBSCAN, Hierarchical Clustering)

## Autoren

Implementiert im Rahmen der Fallstudie "Verbesserung der Vorhersagequalität durch Clustering" für den Kurs "Maschinelles Lernen und Data Mining" 2025 (Jörg Homberger, HFT Stuttgart).

## Lizenz

Dieses Projekt ist Teil einer akademischen Fallstudie.

