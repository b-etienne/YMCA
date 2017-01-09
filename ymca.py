# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:33:29 2016

@author: betienne
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.extmath import randomized_svd


class MCA(object):

    def __init__(self, X, ind=None, supp=None, n_components=None):
        """
        Starts a MCA

        Params:
            X : disjunctive array of size (n,m)
            ncols : number of categorical variables
            n_components : number of axis to keep (non-zero eigenvalues)
                            If None, it is equal to min(n-1,m-ncol)
            supp : supplementary variables, will not be taken into the computation but can
                    be represented.
        """
        self.ncols = X.shape[1]
        if supp:
            self.X_supp = pd.get_dummies(X[supp])
            self.columns_supp = list(self.X_supp.columns)
            self.X_supp = self.X_supp.values
            self.m_supp = self.X_supp.shape[1]

        self.supp = True if supp else False

        if ind:
            X = pd.get_dummies(X[ind])
        else:
            X = pd.get_dummies(X)


        self.index = list(X.index)
        self.columns = list(X.columns)
        self.K = np.matrix(X.values, dtype=np.int8)
        del X
        self.n = self.K.shape[0]
        self.m = self.K.shape[1]
        self.eff_tot = self.K.sum()
        self.n_components = n_components if n_components else min(self.n - 1,self.m - self.ncols)


        print("Individus     : ",self.n)
        print("Variables     : ",self.ncols)
        print(" - Actives    : ", len(ind) if ind else self.ncols)
        print(" - Supp.      : ", len(supp) if supp else 0)
        print("Modalités     : ", self.m + (self.m_supp if supp else 0))
        print(" - Actives    : ", self.m)
        print(" - Supp.      : ", self.m_supp if supp else 0)
        print("Nombre d'axes : ",self.n_components)


        # Inviduals profile weight vector
        self.r = np.array(np.ones([self.n]) / self.n)

        # Variables profile weight vector
        self.c = np.array(self.K.sum(axis=0)).flatten() / (self.n* self.ncols)

        # Frequency Matrix - centered
        self.F = self.K / self.eff_tot - np.outer(self.r, self.c)
        del self.K
        del self.r

        # Dc Matrix
        self.D_c = np.diag(1. / np.sqrt(self.c)) # We invert D_c straightaway.

        # Compute R = UΛVt
        self.R = np.dot(self.F, self.D_c)*(np.sqrt(self.n))

        # Compute SVD
        self.U, self.sigma, self.V = randomized_svd(self.R,
                                                    n_components=self.n_components,
                                                    n_iter=5,random_state=None)
        # Eigen values
        self.eigenvals = self.sigma**2



    def row_profile_matrix(self):
        """
        Returns row profile matrix.
        Warning : for n very large, can cause MemoryError
        Deactivated for the moment.
        """
        pass
#        return(np.linalg.solve(self.D_r, self.F))


    def column_profile_matrix(self):
        """
        Returns column profile matrix
        """
        return(np.linalg.solve(self.D_c, self.F))


    def individuals_coords(self):
        """
        Returns individuals' coords on axis
        """
        return(np.sqrt(self.n) * np.dot(self.U, np.diag(self.sigma)))


    def modalities_coords(self):
        """
        Returns modalities' coords on axis
        """
        y1 = np.dot(np.dot(self.D_c,self.V.T), np.diag(self.sigma))
        if self.supp:
            y2 = self.supplement().T
            return(np.vstack((y1, y2)))
        else:
            return(y1)


    def results(self, benzecri=False):
        """
        Prints the results with eigenvalues and cumulative variance
        Benzecri correction can also be used
        Params:
            benzecri : boolean. If True, performs benzecri correction.
        """

        p = self.ncols
        if benzecri:
            eigen = np.where(self.eigenvals > 1./p,((p/(p-1))*(self.eigenvals-(1/p)))**2,0)
        else:
            eigen = self.eigenvals

        s = eigen.sum()
        c = 0
        print()
        print("Correction : {}\n".format("Benzecri" if benzecri else "None"))
        print("{:>18}{:>15}{:>16}".format("eigenvalue","explained var","cumulative var"))
        print("{:>18}{:>15}{:>16}".format("----------","-------------","--------------"))
        for i, k  in enumerate(eigen):
            c += k/s*100
            print("#{:>5}{:12.6f}{:15.5f}{:16.3f}".format("dim"+str(i+1),k,k/s*100,c))


    def coordinates(self, option=None):
        """
        Returns the coordinates of individuals and modalities on the axis

        Params:
            ndim : number of axis

        Output:
            x : individuals coords
            y : modalities coords
        """

        if option:
            choix = str(option)
        else:
            print("MCA Options")
            print("----------------")
            print("Option 1 : Individuals at barycenter of modalities")
            print("Option 2 : Modalities at barycenter of individuals")
            print("Option 3 : Simultaneous\n")
            choix = input("Choix : ")


        ndim = self.n_components+1

        if choix == "1":
            x = self.individuals_coords()
            y = np.dot(self.modalities_coords(), np.diag(1./self.sigma))
        elif choix == "2":
            x = np.dot(self.individuals_coords(), np.diag(1./self.sigma))
            y = self.modalities_coords()
        elif choix == "3":
            x = self.individuals_coords()
            y = self.modalities_coords()
        else:
            raise ValueError("Le choix doit être  1, 2, 3")

        return(x[:,:ndim], y[:,:ndim])


    def contributions(self):
        """
        Returns the contribution of individuals i and modalities s to the axes.

        """
        x, y = self.coordinates(3)
        y = y[:self.m,:]

        i =  (x**2) / (self.eigenvals * self.n) * 100
        s =  (y**2) * self.c[:, np.newaxis] / self.eigenvals * 100

        return(i, s)


    def plot(self, supp=False):
        """
        Projects on a 2-D graph, with or without supplementary variables
        """

        if supp and not self.supp:
            print("No supplementary variables in chosen configuration")

        else:
            x, y = self.coordinates(3)
            y, y_supp = y[:self.m, :], y[self.m:, :]

            if x.shape[0] > 100:
                print("WARNING : Size of individuals is above 100. The plot will be too heavy.\n Exiting")
            else:
                plt.figure(figsize=(11,8))
                plt.scatter(x[:,0] ,x[:,1], marker='^', color='red', label="individus")
                plt.scatter(y[:,0], y[:,1], marker='s', color='blue', label="mod_actives")

                if supp:
                    plt.scatter(y_supp[:,0], y_supp[:,1], marker='o', color='green', label="mod_supp")
                plt.legend(loc="best")
                plt.axhline(0, linestyle="--", color='k')
                plt.axvline(0, linestyle="--", color='k')
                plt.ylabel("Dim 2 ({:.2f}%)".format(self.eigenvals[1]*100))
                plt.xlabel("Dim 1 ({:.2f}%)".format(self.eigenvals[0]*100))
                for i,j in enumerate(self.index):
                    plt.annotate(j, (x[i,0], x[i,1]))
                for i,j in enumerate(self.columns):
                    plt.annotate(j, (y[i,0], y[i,1]))
                if supp:
                    for i,j in enumerate(self.columns_supp):
                        plt.annotate(j, (y_supp[i,0], y_supp[i,1]))


    def supplement(self):
        """
        Projects the supp variables on the factorial axis as barycenter of the individuals 
        to which they belong
        
        """

#        x_supp = np.zeros((self.n_components,self.X_supp.shape[1]))
#        for c in range(self.X_supp.shape[1]):
#            x_supp[:,c] = np.dot(self.individuals_coords().T,self.X_supp[:,c]) / np.sum(self.X_supp[:,c])
#        return(x_supp)
        return(np.dot(self.individuals_coords().T,self.X_supp) / np.sum(self.X_supp, axis=0))


if __name__ == '__main__':

    df = pd.read_excel('../chiens.xlsx') # Change file name
    df = df.set_index('race') #These are your individuals
    mmca = MCA(df, list(df.columns)[:-2],['affect','agress']) # Choose which columns you want to keep
#    mmca = MCA(df, list(df.columns))
    mmca.results() # Show results
    mmca.plot(True) # Plot results

