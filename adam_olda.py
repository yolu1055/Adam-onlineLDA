import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi
import parse_document
from scipy import linalg as LA
from math import isnan, isinf
import parse_document



n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def argmedian(array):
    indices = n.argsort(array)
    k = indices[len(array) / 2]
    return k

def argone_sixteenth(array):
    indices = n.argsort(array)
    L = len(array)
    indexlist = list()
    indexlist.append(indices[L-1])
    for i in range(15, 7, -1):
        k = indices[(L * i) / 16]
        indexlist.append(k)
    return indexlist




class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, tau0, kappa):
        """
        LDA prarmeters:
        K: #topics
        D: #documents
        W: #vocabulary
        eta: Dirichlet parameter
        alpha: Dirichlet parameter
 
        """
        self._vocab = vocab
        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0
        self._kappa = kappa
        self._updatect = 0
        
        
        

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
       
        #initialize adaptive learning rate
        #(self._g,self._h)=self.initialize_moving_average()

        """
        Adam parameters
        """
        
        self._m = n.zeros(self._lambda.shape)
        self._v = n.zeros(self._lambda.shape)
        self._hat_m = n.ones(self._lambda.shape)
        self._hat_v = n.ones(self._lambda.shape)
        
        self._rhot = 10.0
        self._epsilon = n.ones(self._lambda.shape)  * self._rhot * 100.0
        self._beta_1 = 0.9
        self._beta_2 = 0.999
        
        

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        #(wordids, wordcts) = parse_doc_list(docs, self._vocab)
        (wordids,wordcts)=parse_document.parse_doc_list(docs, self._vocab)
        batchD = len(docs)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             print oldphinorm
#             print n.log(phinorm)
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)


    def do_e_step(self, wordids,wordcts, _gamma, _Elogtheta, _expElogtheta):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        wordids: Given a mini-batch of documents.
                 Each wordids[i] contains word ids in the vocabulary
                 for document i
        wordcts: The corresponding number of words of word wordids[i][n]
        _gamma: q(\theta|\gamma)
        _Elogtheta: E[log(\theta)]
        _expElogtheta: exp{E[log(\theta)]}

  
        """
        
        
        
        batchD=len(wordids)
 
        
        gamma=_gamma
        Elogtheta=_Elogtheta
        expElogtheta=_expElogtheta
        
        sstats = n.zeros(self._lambda.shape)#Return a new array of given shape and type, filled with zeros.
        
        aver_phi = n.zeros(self._lambda.shape)
        
        
        
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
 
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
 
            
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                
                
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                
                
                   
                # If gamma hasn't changed much, we're done.

                
                meanchange = n.mean(abs(gammad - lastgamma))
                
                
 
                
                if (meanchange < meanchangethresh):
                    
                    
                    break
                
            gamma[d, :] = gammad
 
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)
            temp = n.ones(len(cts))
            #aver_phi[:,ids] += n.outer(expElogthetad.T, temp/phinorm)
            #sstats[:,ids]+=_phi.T*cts/phinorm

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta
        #aver_phi = aver_phi * self._expElogbeta
        #sstats_low_order=sstats_low_order*self._expElogbeta
        
        #return(gamma, sstats, aver_phi / batchD)
        return(gamma, sstats)
    
    def test_Adam(self, docs):
        
        (wordids,wordcts)=parse_document.parse_doc_list(docs, self._vocab)
        
        batchD=len(wordids)
        gamma= 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        (gamma, sstats) = self.do_e_step(wordids,wordcts, gamma, Elogtheta, expElogtheta)
        
        bound = self.approx_bound(docs, gamma)
        
        
        docs_Len=len(docs)
        lambda_hat = self._eta+self._D*sstats/docs_Len
        gradient = -self._lambda+ lambda_hat
        
        gradient1 = pow(gradient, 2)
        self._m = self._m * self._beta_1 + (1 - self._beta_1) * gradient
        self._v = self._v * self._beta_2 + (1 - self._beta_2) * gradient1
        self._hat_m = self._m / (1 - pow(self._beta_1, (self._updatect + 1)))
        self._hat_v = self._v / (1 - pow(self._beta_2, (self._updatect + 1)))
        learning_rate = self._rhot / (pow(self._hat_v, 0.5) + self._epsilon)
           
        self._lambda = self._lambda + learning_rate * self._hat_m 
        
        
        self._Elogbeta = dirichlet_expectation(self._lambda) + 1e-10
        
        self._expElogbeta = n.exp(self._Elogbeta)
                    
        self._rhot_max=max(learning_rate.flat[:])
        
        self._rhot_min=min(learning_rate.flat[:])
        
        
        self._updatect += 1
        return (gamma, bound)
    
    
    def save_specific_parameter(self, word_list, gradient, lr):
        for v in word_list:
            f1 = open("./lam_%d.txt"%v, "a")
            f2 = open("./g_%d.txt"%v, "a")
            f3 = open("./lr_%d.txt"%v, "a")
            for k in range(0, self._K):
                f1.write("%e"%self._lambda[k][v] + "\t")
                f2.write("%e"%gradient[k][v] + "\t")
                f3.write("%e"%lr[k][v] + "\t")
            f1.write("\n")
            f2.write("\n")
            f3.write("\n")
            f1.close()
            f2.close()
            f3.close()
                     
