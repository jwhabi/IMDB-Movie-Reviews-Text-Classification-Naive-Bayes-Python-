import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

class Eval:
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold
        #print(self.pred)
        
    def Accuracy(self):
       
        return np.sum(np.equal(self.pred, self.gold)) / float(len(self.gold))
    
    def Recall(self):
        return recall_score(self.gold,self.pred)
    
    def Precision(self):
         return precision_score(self.gold,self.pred)
     
    def PvRcurve(self):
        average_precision = average_precision_score(self.gold, self.pred)
        precision, recall, _ = precision_recall_curve(self.gold, self.pred)

        plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
        plt.show()
    
