class Metrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.precision = None
        self.recall = None
        self.f1_score = None

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self._precision = None
        self._recall = None
        self._f1_score = None
    
    def update(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    @property
    def precision(self):
        if self._precision is None:
            self._precision = self.tp / (self.tp + self.fp)
        
        return self._precision

    @property    
    def recall(self):
        if self._recall is None:
            self._recall = self.tp / (self.tp + self.fn)

        return self._recall
    
    @property
    def f1_score(self):
        if self._f1_score is None:
            self._f1_score = (2 * self.precision * self.recall) / (self.precision + self.recall)

        return self._f1_score

    @precision.setter
    def precision(self, value):
        self._precision = value
    
    @recall.setter
    def recall(self, value):
        self._recall = value

    @f1_score.setter
    def f1_score(self, value):
        self._f1_score = value

  


class MultiClassMetrics:

    def __init__(self):
        self.class_metrics = []

    def update(self, confusion_matrix):
        num_of_classes = len(confusion_matrix)
        for i in range(num_of_classes):
            self.class_metrics.append(Metrics())

        for i in range(num_of_classes):
            fp = 0
            fn = 0
            tp = confusion_matrix[i][i]
            predicted = confusion_matrix[:, i]
            true = confusion_matrix[i, :]

            if i < num_of_classes:
                fp += sum(predicted[i+1:])
                fn += sum(true[i+1:])
            if i > 0:
                fp += sum(predicted[:i])
                fn += sum(true[:i])

            metric = self.class_metrics[i]
            metric.update(tp, fp, 0, fn)