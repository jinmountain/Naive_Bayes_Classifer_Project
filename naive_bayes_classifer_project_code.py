import numpy
import scipy.io
import math
import geneNewData

import numpy as np
class NaiveBayes:
    def parameterPrep(self, train0, train1):
        self.feature1_train0 = np.zeros(len(train0), dtype=np.float64)
        self.feature2_train0 = np.zeros(len(train0), dtype=np.float64)

        self.feature1_train1 = np.zeros(len(train1), dtype=np.float64)
        self.feature2_train1 = np.zeros(len(train1), dtype=np.float64)

        for i in range(len(train0)):
            self.feature1_train0[i] = train0[i].mean()
            self.feature2_train0[i] = train0[i].std()

        for i in range(len(train1)):
            self.feature1_train1[i] = train1[i].mean()
            self.feature2_train1[i] = train1[i].std()

        mean_ft1_train0 = self.feature1_train0.mean()
        var_ft1_train0 = self.feature1_train0.var()
        mean_ft2_train0 = self.feature2_train0.mean()
        var_ft2_train0 = self.feature2_train0.var()

        mean_ft1_train1 = self.feature1_train1.mean()
        var_ft1_train1 = self.feature1_train1.var()
        mean_ft2_train1 = self.feature2_train1.mean()
        var_ft2_train1 = self.feature2_train1.var()
        
        self.train0_prior = len(train0)/(len(train0) + len(train1))
        self.train1_prior = len(train1)/(len(train0) + len(train1))

        self.parameters = [mean_ft1_train0, var_ft1_train0, mean_ft2_train0, var_ft2_train0, mean_ft1_train1, var_ft1_train1, mean_ft2_train1, var_ft2_train1]

    def pdf(self, feature1, feature2, ft1_mean, ft1_var, ft2_mean, ft2_var):
        
        numerator = np.exp(-(feature1 - ft1_mean)**2 / (2 * ft1_var))
        denominator = np.sqrt(2 * np.pi * ft1_var)
        ft1_pdf = numerator/denominator

        numerator = np.exp(-(feature2 - ft2_mean)**2 / (2 * ft2_var))
        denominator = np.sqrt(2 * np.pi * ft2_var)
        ft2_pdf = numerator/denominator

        ft_comb_pdf = ft1_pdf * ft2_pdf
        return ft_comb_pdf 

    def predict(self, test0, test1, train_parameters):
        feature1_t0 = np.zeros(len(test0), dtype=np.float64)
        feature2_t0 = np.zeros(len(test0), dtype=np.float64)
        
        feature1_t1 = np.zeros(len(test1), dtype=np.float64)
        feature2_t1 = np.zeros(len(test1), dtype=np.float64)
        
        for i in range(len(test0)):
            feature1_t0[i] = test0[i].mean()
            feature2_t0[i] = test0[i].std()

        for i in range(len(test1)):
            feature1_t1[i] = test1[i].mean()
            feature2_t1[i] = test1[i].std()
            
        #train1 and train2 mean and var of feature1 and feature2
        mean_ft1_train0 = train_parameters[0]
        var_ft1_train0 = train_parameters[1]
        mean_ft2_train0 = train_parameters[2]
        var_ft2_train0 = train_parameters[3]
        
        mean_ft1_train1 = train_parameters[4]
        var_ft1_train1 = train_parameters[5]
        mean_ft2_train1= train_parameters[6]
        var_ft2_train1= train_parameters[7]
        
        print("Parameters: ", self.parameters)
        
        test0_true = 0
        test0_false = 0
        for i in range(len(test0)):
            # with train0 mean and var of ft1 and ft2
            test0_train0_pdf = self.pdf(feature1_t0[i], feature2_t0[i], mean_ft1_train0, var_ft1_train0, mean_ft2_train0, var_ft2_train0)
            test0_train0_prior = self.train0_prior
            test0_train0_posterior = np.log(test0_train0_pdf) + np.log(test0_train0_prior)
            
            test0_train1_pdf = self.pdf(feature1_t0[i], feature2_t0[i], mean_ft1_train1, var_ft1_train1, mean_ft2_train1, var_ft2_train1)
            test0_train1_prior = self.train1_prior
            test0_train1_posterior = np.log(test0_train1_pdf) + np.log(test0_train1_prior)
            
            if test0_train0_posterior > test0_train1_posterior:
                test0_true += 1
            else:
                test0_false += 1
        test0_accuracy = test0_true/(len(test0))
        print("accuray for test0: ", test0_accuracy)
        
        test1_true = 0
        test1_false = 0
        for i in range(len(test1)):
            # with train0 mean and var of ft1 and ft2
            test1_train0_pdf = self.pdf(feature1_t1[i], feature2_t1[i], mean_ft1_train0, var_ft1_train0, mean_ft2_train0, var_ft2_train0)
            test1_train0_prior = self.train0_prior
            test1_train0_posterior = np.log(test1_train0_pdf) + np.log(test1_train0_prior)
            
            test1_train1_pdf = self.pdf(feature1_t1[i], feature2_t1[i], mean_ft1_train1, var_ft1_train1, mean_ft2_train1, var_ft2_train1)
            test1_train1_prior = self.train1_prior
            test1_train1_posterior = np.log(test1_train1_pdf) + np.log(test1_train0_prior)
            
            if test1_train1_posterior > test1_train0_posterior:
                test1_true += 1
            else:
                test1_false += 1
        test1_accuracy = test1_true/(len(test1))
        print("accuray for test1: ", test1_accuracy)
            
def main():
    myID='0233'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    
    print([len(train0),len(train1),len(test0),len(test1)])
    nb = NaiveBayes()
    nb.parameterPrep(train0, train1)
    nb.predict(test0, test1, nb.parameters)
    
if __name__ == '__main__':
    main()