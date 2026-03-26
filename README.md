# Del, First Meeting

## Sources
Dataset:   The Street View House Numbers (SVHN) Dataset
- http://ufldl.stanford.edu/housenumbers/
Archidecture:
Inspired by:
- Handwritten Digit Recognition with a Back-Propagation Network (https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)

## Proposed archidecture:
* Input: 32x32x3
* Conv1: 16 Filter, 5x5 Kernel, Stride 1, Relu
* Conv2: 32 Filter, 5x5 Kernel, Stride 1, ReLU
Max Pooling
* Conv3: 128 Filter, 5x5 Kernel, Stride 1, ReLU
Max Pooling
* Flatten
* FC1: 512 Neuronen, Relu
* FC2: 96 Neuronen,Relu
* Output: 10 Neuronen, Softmax






Daten Kennenlernen, Aufbau Modellierung, Basismodell
    1. Vorbereitung der Daten:
        a. (kurze) explorative Analyse der Features (z.B. Vergleich der Klassen pro Feature, Balanciertheit der Klassen).
        b. geeignetes Preprocessing (z.B. Normalisierung der Daten)
    2. Festlegen der Metriken für die Evaluation.
    3. Implementiere Basisfunktionalität:
        a. Fürs Trainieren und Evaluieren von Modellen.
        b. Verwende MLOps Plattform (z.B. W&B)
        c. Implementiere das vereinbarte Basismodell.
        d. Overfitting Test: Trainiere das Basismodell mit einem einzigen Sample oder einem Batch, um die Basisfunktionalität zu testen.
    4. Training des Basismodells:
        a. Training mit SGD (ohne Momentum), ohne Regularisierung, ohne Batchnorm unter Verwendung von allen Trainingsdaten;
        b. Tuning der Lernrate und der Batchsize. Zeige die Unterschiede anhand der Lernkurven mit der gewählten Kostenfunktion und den Metriken auf. Diskussion der Ergebnisse und Vergleich mit der Erwartung aus der Theorie.
        Bemerkung: Keine Verfahren zur automatischen Hyperparameter-Suche (z.B. keine Bayes'sche Optimierung) verwenden!
        c. Schätzung des statistischen Fehlers dieser Metriken (mit cross-validation, zumindest für ein Training durchgeführt).