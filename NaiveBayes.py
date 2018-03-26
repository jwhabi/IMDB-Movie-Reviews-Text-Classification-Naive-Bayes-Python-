import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
         self.ALPHA = ALPHA
         self.data = data # training data
        #TODO: Initalize parameters - Random initialization
         self.vocab_len = 1
         self.count_positive = 1
         self.count_negative = 1
         self.num_positive_reviews = 1
         self.num_negative_reviews = 1
         self.total_positive_words = 1
         self.total_negative_words = 1
         self.P_positive = 1
         self.P_negative = 1
         self.deno_pos = 1
         self.deno_neg =1
         self.pos_words=[]
         self.neg_words=[]
         self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        #calculate frequency parameters to be used in PredictLabel to calculate probabilities
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        
        self.num_positive_reviews = sum([1 if i==1 else 0 for i in Y])
        self.num_negative_reviews = sum([1 if i==-1 else 0 for i in Y])
        self.count_positive = np.zeros(X.shape[1])
        self.count_negative = np.zeros(X.shape[1])
        self.total_positive_words = np.sum(X[positive_indices,:])
        self.total_negative_words = np.sum(X[negative_indices,:])
        
        #For smoothing
        self.deno_pos = self.total_positive_words + self.ALPHA*X.shape[1]
        self.deno_neg = self.total_negative_words + self.ALPHA*X.shape[1]
        
        rows,columns = X.nonzero()
        for i,j in zip(rows,columns):
            if self.data.Y[i]==1:
                    self.count_positive[j]+=X[i,j]
            else:
                    self.count_negative[j]+=X[i,j]
        self.count_positive = (self.count_positive + self.ALPHA)
        self.count_negative = (self.count_negative + self.ALPHA)        
        #above 2 arrays give total frequencies for each word in each class
        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X,threshold=0.5):
        #TODO: Implement Naive Bayes Classification
        #Calculate P(W|C) and P(C) to get P(C|W) by doing a logsum
        self.P_positive = log(self.num_positive_reviews)-(log(self.num_positive_reviews)+log(self.num_negative_reviews))
        self.P_negative = log(self.num_negative_reviews)-(log(self.num_positive_reviews)+log(self.num_negative_reviews))
        pred_labels = []
        w=X.shape[1]
        sh = X.shape[0]
        for i in range(sh):
           #checks if the value of the data is zero or not if not then proceed
            z = X[i].nonzero()
            positive_sum = self.P_positive
            negative_sum = self.P_negative
            for j in range(len(z[0])):
                # Look at each feature

                row_index = i
                col_index = z[1][j]
                occurrence = X[row_index, col_index]
                P_pos = log(self.count_positive[col_index]) - log(self.deno_pos)
                positive_sum = positive_sum + occurrence * P_pos
                P_neg = log(self.count_negative[col_index]) - log(self.deno_neg)
                negative_sum = negative_sum + occurrence * P_neg
            probValue = exp(positive_sum - self.LogSum(positive_sum, negative_sum))
            if probValue > threshold:            # Predict positive
                pred_labels.append(1.0)
            else:               # Predict negative
                pred_labels.append(-1.0)  
            
#            if positive_sum > negative_sum:
#
#                pred_labels.append(1.0)
#
#            else:
#
#                pred_labels.append(-1.0)

              

        return pred_labels


    def LogSum(self, logx, logy):   
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)   
        #print(m)
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test,indexes):
         
         for i in indexes:

            # TO DO: Predict the probability of the i_th review in test being positive review

            # TO DO: Use the LogSum function to avoid underflow/overflow

            predicted_label = 0
            z = test.X[i].nonzero()
            positive_sum = self.P_positive
            negative_sum = self.P_negative

            for j in range(len(z[0])):

                row_index = i
                col_index = z[1][j]
                occurrence = test.X[row_index, col_index]
                P_pos = log(self.count_positive[col_index])
                positive_sum = positive_sum + occurrence * P_pos
                P_neg = log(self.count_negative[col_index])
                negative_sum = negative_sum + occurrence * P_neg
                

            predicted_prob_positive = exp(positive_sum - self.LogSum(positive_sum, negative_sum))

            predicted_prob_negative = exp(negative_sum - self.LogSum(positive_sum, negative_sum))

           

            if positive_sum > negative_sum:

                predicted_label=1.0

            else:

                predicted_label=-1.0

               

            print (test.Y[i], test.X_reviews[i],predicted_label)

            # TO DO: Comment the line above, and uncomment the line below

            #print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative)

    # Evaluate performance on test data 
    #Gives Accuracy, Precision and Recall for different values of a threshold
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        print("For Positive Class:")
        print("Test Accuracy: ",ev.Accuracy())
        print("Test Recall: ",ev.Recall())
        print("Test Precision: ",ev.Precision())
        print("\n")
        print("For Negative Class:")
        ev_neg = Eval([1 if i == -1 else -1 for i in Y_pred], [1 if i == -1 else -1 for i in test.Y])
        print("Test Accuracy: ",ev_neg.Accuracy())
        print("Test Recall: ",ev_neg.Recall())
        print("Test Precision: ",ev_neg.Precision())
        probality_threshold=[0.2,0.4,0.6,0.8]
        Precision=[]
        Recall=[]
        Precision.append(ev.Precision())
        Recall.append(ev.Recall())
        length=len(probality_threshold)
        for i in range(0,length):
            Y_pred = self.PredictLabel(test.X,probality_threshold[i])
            ev = Eval(Y_pred, test.Y)
            Precision.append(ev.Precision())
            Recall.append(ev.Recall())
        plt.plot(Precision,Recall)
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve')
        plt.show()
        #above statements show a Precision vs Recall graph

    #TO-DO: give top 20 positive and negative words based on their log odds
    #words with the max difference value would be part of this list
    def Features(self):
        pos_diff=np.zeros(self.data.X.shape[1])
        neg_diff=np.zeros(self.data.X.shape[1])
        for j in range(len(self.count_positive)):
                P_pos = log(self.count_positive[j]) - log(self.deno_pos)
                P_neg = log(self.count_negative[j]) - log(self.deno_neg)
                pos_diff[j]=(P_pos-P_neg)
                #neg_diff[j]=(P_neg-P_pos)*(self.count_negative[j]-self.count_positive[j])
        print("Top 20 Positive words with their weights:")        
        pos_index=pos_diff.argsort()[-20:][::-1]
        for j in pos_index:
            print("j:",self.data.vocab.GetWord(j)," ",pos_diff[j])
        
        print("Top 20 Negative words with their weights:")
        neg_index=pos_diff.argsort()[:20]
        for j in neg_index:
            print("j:",self.data.vocab.GetWord(j)," ",pos_diff[j])
            #Alternate better approach below: needs dubugging
        
#                pos_diff=np.zeros(self.data.X.shape[1])
#        neg_diff=np.zeros(self.data.X.shape[1])
#        for j in range(len(self.count_positive)):
#                P_pos = log(self.count_positive[j]) - log(self.deno_pos)
#                P_neg = log(self.count_negative[j]) - log(self.deno_neg)
#                pos_diff[j]=(P_pos-P_neg)*(self.count_positive[j]-self.count_negative[j])
#                neg_diff[j]=(P_neg-P_pos)*(self.count_negative[j]-self.count_positive[j])
#        print("Top 20 Positive words with their weights:")        
#        pos_index=pos_diff.argsort()[-20:][::-1]
#        for j in pos_index:
#            print("j:",self.data.vocab.GetWord(j)," ",pos_diff[j])
#        
#        print("Top 20 Negative words with their weights:")
#        neg_index=neg_diff.argsort()[:20]
#        for j in neg_index:
#            print("j:",self.data.vocab.GetWord(j)," ",neg_diff[j])    

if __name__ == "__main__":
    print(sys.argv)
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    nb.Eval(testdata)
    nb.PredictProb(testdata,range(10))
    nb.Features()


