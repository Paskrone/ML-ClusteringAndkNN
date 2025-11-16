import java.io.File;
import java.io.IOException;
import java.util.*;


public class HauptprogrammVL {

	public static void main(String[] args) throws IOException {

		/*
		 * NN_Visualisierung_ZWEI_KLASSEN();
		 * KNearestNeigbour_Visualisierung_ZWEI_KLASSEN();
		 * DecisionTree_Visualisierung_ZWEI_KLASSEN();
		 * 
		 * NN_Diabetes(); KNearestNeigbour_Diabetes(); DecisionTreeDiabetes();
		 * KMeans_Diabetes();
		 */
		
		// Fallstudie: Clustering zur Verbesserung der Vorhersagequalität
		// Alle 4 Durchläufe ausführen und vergleichen
		compareAllApproaches();
		

	}

	public static void DecisionTree_Visualisierung_ZWEI_KLASSEN() {
		// 1 Daten Einlesen
		// double daten[][] = Einlesen.einlesenXY(new File("data\\x_vs_cluster.csv"));
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\xor.csv"));
//		double daten[][] = Einlesen.einlesenXY(new File("data\\linear.csv"));
		double daten[][] = Einlesen.einlesenXY(new File("data\\clusters.csv"));
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\circle.csv"));
//		double[][] daten    = {{10,10,0},{0,0,0},{10,0,1},{0,10,1}, {5, 5, 0}, {3, 3, 1}}; 

		double[][] features = Einlesen.features(daten, Einlesen.ScalerType.MINMAX);
		double[][] labels = Einlesen.labelsBinary(daten);

		int maxDepth = 7;
		DecisionTree dt = new DecisionTree(features, labels, maxDepth);
		dt.buildTree();

		DecisionTree.DTSolver dtSolver = new DecisionTree.DTSolver(dt);
		VisualizerUniversal.showGui(dtSolver, features, labels, 2);

	}

	public static void KNearestNeigbour_Visualisierung_ZWEI_KLASSEN() {

		// 1 Daten Einlesen
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\x_vs_cluster.csv"));
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\xor.csv"));
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\linear.csv"));
		double daten[][] = Einlesen.einlesenXY(new File("data\\clusters.csv"));
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\circle.csv"));
//		double[][] daten    = {{10,10,0},{0,0,0},{10,0,1},{0,10,1}, {5, 5, 0}, {3, 3, 1}}; 

		double[][] features = Einlesen.features(daten, Einlesen.ScalerType.MINMAX);
		double[][] labels = Einlesen.labelsBinary(daten);

		// kNN initialisieren
		KNearestNeighbor knn2 = new KNearestNeighbor(features, labels, 1);

		// GUI starten
		// KNNVisualizer.showGui(knn2, features, labels);
		KNearestNeighbor.KNNSolver knnSolver = new KNearestNeighbor.KNNSolver(knn2);
		VisualizerUniversal.showGui(knnSolver, features, labels, 2);

	}

	public static void KNearestNeigbour_Diabetes() {

		double[][] daten = Einlesen.einlesenDiabetes(new File("data\\diabetes.csv"));
		double[][] features = Einlesen.features(daten);
		double[][] labels = Einlesen.labelsBinary(daten);

		// 1. Splitten
		double[][][] sets = new double[4][][];
		double trainFraction = 0.8;
		long seed = 886L;

		Einlesen.splitBinary(features, labels, sets, trainFraction, seed);
		double[][] trainFeatures = sets[0];
		double[][] testFeatures = sets[1];
		double[][] trainLabels = sets[2];
		double[][] testLabels = sets[3];

		// 2. Skalierer fitten und anwenden
		MinMaxScaler scaler = new MinMaxScaler();
		scaler.fit(trainFeatures);
		trainFeatures = scaler.transform(trainFeatures);
		testFeatures = scaler.transform(testFeatures);

		// 3. kNN-Modell erzeugen und trainieren (d.h. hier Trainingsdaten dauerhaft
		// speichern)
		int k = 55;
		KNearestNeighbor knn = new KNearestNeighbor(trainFeatures, trainLabels, k);
		//KNN_FAST knn = new KNN_FAST(trainFeatures, trainLabels, k);

		// 4. Evaluation
		double accuracy = knn.analyze(testFeatures, testLabels);
		System.out.println("KNN Genauigkeit: " + accuracy);

	}

	
	public static void DecisionTreeDiabetes() {
	
		double[][] daten    = Einlesen.einlesenDiabetes(new File("data\\diabetes.csv"));
		double[][] features = Einlesen.features(daten);
		double[][] labels   = Einlesen.labelsBinary(daten);

		// 1. Splitten
		double[][][] sets = new double[4][][];
		double trainFraction = 0.8;
		long seed = 86L;

		Einlesen.splitBinary(features, labels, sets, trainFraction, seed);
		double[][] trainFeatures = sets[0];
		double[][] testFeatures  = sets[1];
		double[][] trainLabels   = sets[2];
		double[][] testLabels    = sets[3];

		// 2. Skalierer fitten und anwenden
		MinMaxScaler scaler = new MinMaxScaler();
		scaler.fit(trainFeatures);
		trainFeatures = scaler.transform(trainFeatures);
		testFeatures = scaler.transform(testFeatures);

		
		int maxDepth = 3;
		DecisionTree dt = new DecisionTree(trainFeatures, trainLabels, maxDepth);
		dt.buildTree();
		double dtAccuracy = dt.analyze(testFeatures, testLabels);
		System.out.println("Decision Tree Genauigkeit: " + dtAccuracy);
	}

	
	

	public static void NN_Visualisierung_ZWEI_KLASSEN() {
 //     double daten[][]    = Einlesen.einlesenXY(new File("data\\x_vs_cluster.csv"));
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\xor.csv"));
//		double daten[][] = Einlesen.einlesenXY(new File("data\\linear.csv"));
		double daten[][]    = Einlesen.einlesenXY(new File("data\\clusters.csv"));
//		double daten[][]    = Einlesen.einlesenXY(new File("data\\circle.csv"));	
//		double[][] daten    = {{10,10,0},{1,1,1},{5,5,0},{7,3,1}}; 
//		double[][] daten    = {{10,10,0},{1,1,1},{5,5,0},{7,3,1},{8,4,0},{6,6,1}}; 


		double[][] trainFeatures = Einlesen.features(daten, Einlesen.ScalerType.MINMAX);
		double[][] trainLabels   = Einlesen.labelsBinary(daten);

		// 2. Netz aufbauen
		int inputSize             = trainFeatures[0].length;
		int outputSize            = trainLabels[0].length;	
		int[] layerSizes          = {inputSize, 50, outputSize};
		String hiddenActivations  = "sigm"; // "relu", "sigm"
		String outputActivation   = "sigm";
		String taskType           = "binary";
		long seedNet              = 886L;
		FFN net = new FFN(layerSizes, hiddenActivations, outputActivation, taskType, seedNet);

		// 3. Training
		int epochs                = 2000; 
		double learningRate       = 0.5; 
		LossFunction lossFunction = new BinaryCrossEntropy(); //MeamSquaredError
		net.train(trainFeatures, trainLabels, epochs, learningRate, lossFunction);
		
		// 4. Anwendung auf den gesamten Featureraum
		FFNSolver ffnSolver = new FFNSolver(net);
		VisualizerUniversal.showGui(ffnSolver, trainFeatures, trainLabels, 2);
		
	}

	
	
	
	public static void NN_Diabetes() {
		double[][] daten    = Einlesen.einlesenDiabetes(new File("data\\diabetes.csv"));
		double[][] features = Einlesen.features(daten);
		double[][] labels   = Einlesen.labelsBinary(daten);



		// 1. Splitten
		double[][][] sets = new double[4][][];
		double trainFraction = 0.8;
		long seedSplit      = 878L;
		Einlesen.splitBinary(features, labels, sets, trainFraction, seedSplit);
		double[][] trainFeatures = sets[0];
		double[][] testFeatures  = sets[1];
		double[][] trainLabels   = sets[2];
		double[][] testLabels    = sets[3];

		// 2. Skalierer 
		MinMaxScaler scaler = new MinMaxScaler();
		scaler.fit(trainFeatures);
		trainFeatures = scaler.transform(trainFeatures);
		testFeatures  = scaler.transform(testFeatures);

		// 2. Netz aufbauen
		int inputSize             = features[0].length;
		int outputSize            = labels[0].length;	
		int[] layerSizes          = {inputSize, 10, outputSize};
		String hiddenActivations  = "sigm"; // "relu", "sigm"
		String outputActivation   = "sigm";
		String taskType           = "binary";
		long seedNet              = 50;
		
		
		FFN net = new FFN(layerSizes, hiddenActivations, outputActivation, taskType, seedNet);

		// 3. Training
		int epochs                = 1000; 
		double learningRate       = 0.01; 
		LossFunction lossFunction = new MeanSquaredError(); 
		net.train(trainFeatures, trainLabels, epochs, learningRate, lossFunction);

		// 4. Test
		net.validate(testFeatures, testLabels);	
	}
	
	public static void KMeans_Diabetes() {
	    // 1. Daten laden (ueber Einlesen.java)
	    double[][] daten = Einlesen.einlesenDiabetes(new File("data/diabetes_scaled_train.csv"));

	    // 2. In List<DataPoint> konvertieren
	    List<DataPoint> data = new ArrayList<>();
	    for (double[] row : daten) {
	        double[] features = Arrays.copyOf(row, row.length - 1); // alles ausser letzte Spalte
	        int label = (int) row[row.length - 1];                  // letzte Spalte als Label
	        data.add(new DataPoint(features, label));
	    }

	    // 3. K-Means mit z. B. 4 Clustern
	    KMeans km = new KMeans(data, 4);
	    km.run();

	    // 4. Ergebnisse anzeigen
	    System.out.println("KMeans: k=" + 4);
	    km.printCentroids();
	}
	
	/**
	 * Baseline-KNN ohne Clustering
	 * Durchlauf 1: Direkt KNN auf skalierten Daten
	 */
	public static void baselineKNN_Diabetes() {
		int[] kValues = {5, 10, 15, 20, 25, 30, 55};
		File trainFile = new File("data/diabetes_scaled_train.csv");
		File testFile = new File("data/diabetes_scaled_test.csv");
		
		BaselinePipeline.BaselineResult result = BaselinePipeline.runBaseline(trainFile, testFile, kValues);
		
		System.out.println("\n=== BASELINE ABGESCHLOSSEN ===");
		System.out.println("Bester k-Wert: " + result.k);
		System.out.println("Finale Accuracy: " + String.format("%.4f", result.metrics.getAccuracy()));
	}
	
	/**
	 * Durchlauf 2: Cluster-basierte Datenimputation
	 * Clustering → Imputation → KNN
	 */
	public static void imputationKNN_Diabetes() {
		int[] kValues = {5, 10, 15, 20, 25, 30, 55};
		int numClusters = 4; // Optimale Clusteranzahl kann später angepasst werden
		File trainFile = new File("data/diabetes_scaled_train.csv");
		File testFile = new File("data/diabetes_scaled_test.csv");
		
		ClusterImputationPipeline.ImputationResult result = 
			ClusterImputationPipeline.runImputation(trainFile, testFile, numClusters, kValues);
		
		System.out.println("\n=== IMPUTATION ABGESCHLOSSEN ===");
		System.out.println("Bester k-Wert: " + result.k);
		System.out.println("Anzahl Cluster: " + result.numClusters);
		System.out.println("Finale Accuracy: " + String.format("%.4f", result.metrics.getAccuracy()));
	}
	
	/**
	 * Durchlauf 3: Feature-Erweiterung mit Cluster-Informationen
	 * Clustering → Feature-Erweiterung → KNN
	 */
	public static void featureExtensionKNN_Diabetes() {
		int[] kValues = {5, 10, 15, 20, 25, 30, 55};
		int numClusters = 4;
		boolean includeDistances = true; // Distanzen zu Clustern als zusätzliche Features
		File trainFile = new File("data/diabetes_scaled_train.csv");
		File testFile = new File("data/diabetes_scaled_test.csv");
		
		ClusterFeatureExtensionPipeline.FeatureExtensionResult result = 
			ClusterFeatureExtensionPipeline.runFeatureExtension(trainFile, testFile, numClusters, kValues, includeDistances);
		
		System.out.println("\n=== FEATURE-ERWEITERUNG ABGESCHLOSSEN ===");
		System.out.println("Bester k-Wert: " + result.k);
		System.out.println("Anzahl Cluster: " + result.numClusters);
		System.out.println("Finale Accuracy: " + String.format("%.4f", result.metrics.getAccuracy()));
	}
	
	/**
	 * Durchlauf 4: Feature-Transformation basierend auf Clustern
	 * Clustering → Feature-Transformation → KNN
	 */
	public static void transformationKNN_Diabetes() {
		int[] kValues = {5, 10, 15, 20, 25, 30, 55};
		int numClusters = 4;
		String transformationType = "cluster_features"; // "cluster_scaled", "pca", "cluster_features"
		File trainFile = new File("data/diabetes_scaled_train.csv");
		File testFile = new File("data/diabetes_scaled_test.csv");
		
		ClusterTransformationPipeline.TransformationResult result = 
			ClusterTransformationPipeline.runTransformation(trainFile, testFile, numClusters, kValues, transformationType);
		
		System.out.println("\n=== TRANSFORMATION ABGESCHLOSSEN ===");
		System.out.println("Bester k-Wert: " + result.k);
		System.out.println("Anzahl Cluster: " + result.numClusters);
		System.out.println("Transformation: " + result.transformationType);
		System.out.println("Finale Accuracy: " + String.format("%.4f", result.metrics.getAccuracy()));
	}
	
	/**
	 * Vergleich aller 4 Durchläufe
	 */
	public static void compareAllApproaches() {
		System.out.println("╔════════════════════════════════════════════════════════════════╗");
		System.out.println("║     VERGLEICH ALLER CLUSTERING-ANSÄTZE                        ║");
		System.out.println("╚════════════════════════════════════════════════════════════════╝\n");
		
		// Alle Durchläufe ausführen
		System.out.println("DURCHLAUF 1: BASELINE (OHNE CLUSTERING)");
		System.out.println("════════════════════════════════════════════════════════════════");
		baselineKNN_Diabetes();
		
		System.out.println("\n\nDURCHLAUF 2: CLUSTER-BASIERTE IMPUTATION");
		System.out.println("════════════════════════════════════════════════════════════════");
		imputationKNN_Diabetes();
		
		System.out.println("\n\nDURCHLAUF 3: FEATURE-ERWEITERUNG");
		System.out.println("════════════════════════════════════════════════════════════════");
		featureExtensionKNN_Diabetes();
		
		System.out.println("\n\nDURCHLAUF 4: FEATURE-TRANSFORMATION");
		System.out.println("════════════════════════════════════════════════════════════════");
		transformationKNN_Diabetes();
		
		System.out.println("\n\n╔════════════════════════════════════════════════════════════════╗");
		System.out.println("║     ALLE DURCHLÄUFE ABGESCHLOSSEN                              ║");
		System.out.println("╚════════════════════════════════════════════════════════════════╝");
	}

	
	


	
	
}
