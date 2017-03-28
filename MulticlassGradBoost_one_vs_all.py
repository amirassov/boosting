from pylab import *
from common.console import Logger
from common.iteration import piter
from common.classes.Struct import Struct
import sklearn as skl
import scipy

class MulticlassGradBoost:
	'''Author: Konstantin Kozlovtsev.'''
    def __init__(self, base_learner, base_learners_count, loss=None,
                 fit_coefs=True, refit_tree=True, shrinkage=1, max_fun_evals=200, xtol=10**-6, ftol=10**-6, log_level=0):
        self.base_learners_count = base_learners_count
        self.base_learner = base_learner
        self.fit_coefs = fit_coefs
        self.shrinkage = shrinkage
        self.log = Logger(log_level)
        self.refit_tree = refit_tree
        self.optimization = Struct(max_fun_evals=max_fun_evals, xtol=xtol, ftol=ftol)
        self.n_classes = None
        self.estimators = []
        self.loss = loss
        self.log_level=log_level

    def fit(self, X, y, X_val=None, y_val=None, bad_iters_count=+inf):
        targets = np.unique(y)
        self.n_classes = targets.shape[0]

        if targets.min() < 0 or targets.max() >= self.n_classes:
            raise Exception('class marks need be in [0, n), for n - number of classes')

        for i in range(self.n_classes):
            y_i = np.zeros(X.shape[0])
            y_i[y == i] = 1
            if X_val is not None and y_val is not None:
                y_i_val = np.zeros(X_val.shape[0])
                y_i_val[y_val == i] = 1
            gb = BinaryGradBoost(
                self.base_learner,
                self.base_learners_count,
                self.loss,
                self.fit_coefs,
                self.refit_tree,
                self.shrinkage,
                self.optimization.max_fun_evals,
                self.optimization.xtol,
                self.optimization.ftol,
                self.log_level
            )
            if X_val is not None and y_val is not None:
                gb.fit(X, y_i, X_val, y_i_val, bad_iters_count)
            else:
                gb.fit(X, y_i)
            self.estimators.append(gb)

    def predict(self, X, max_base_learners=+inf):
        predictions = []
        for gb in self.estimators:
            predictions.append(gb.F(X, max_base_learners))
        predictions = np.array(predictions)
        return predictions.argmax(axis=0)

    def predict_proba(self, X, max_base_learners):
        pass


class BinaryGradBoost:
    '''Realization of gradient boosting predictor. 
    Params:
        base_learner (object): instance of base learner class, initialized with necessary parameters. 
        All base learners inside are recreated with these parameters.
        loss (string): loss function. Possible values are: "square", "exp" and "log".
        base_learners_count (int): how many base learners to fit (number of boosting iterations)
        fit_coefs (boolean): whether to fit or not multiplier coefficients by each base learner 
        refit_tree (boolean): in case the base learner is regression tree, whether to refit or not leaf predictions of each tree.
        shrinkage (float): how much to multiply each coefficient by the base learner
        log_level (int): how many debug messages to display. The lower the value, the more logger messages will be shown.
    Comments:
        - Univarite regression predictions are made for loss="square".
        - Binary classification predictions are made for loss="exp" or loss="log". In these cases y=0 or y=1.
        - For loss="log" not only classes can be predicted (with predict function) but also class probabilities (with predict_proba function)
        - Zero-th approximation is zero. Higher order approximations are sums of base learners with coefficients.
    Author: 
        Victor Kitov, 03.2016.'''

    def __init__(self, base_learner, base_learners_count, loss=None,  
                 fit_coefs=True, refit_tree=True, shrinkage=1, max_fun_evals=200, xtol=10**-6, ftol=10**-6, log_level=0):
        self.base_learners_count = base_learners_count
        self.base_learner = base_learner
        self.fit_coefs = fit_coefs
        self.shrinkage = shrinkage
        self.log = Logger(log_level)
        self.refit_tree = refit_tree
        self.optimization = Struct(max_fun_evals=max_fun_evals, xtol=xtol, ftol=ftol)
        
        self.coefs = []
        self.base_learners = []
        
        if loss=='square':
            self.loss = lambda r,y: 0.5*(r-y)**2
            self.loss_derivative = lambda r, y: (r-y)
            self.task='regression'
        elif loss=='exp':
            self.loss = lambda r,y: exp(-r*y)
            self.loss_derivative = lambda r, y: -(y*exp(-r*y))
            self.task='classification'
        elif loss=='log':
            self.loss = lambda r,y: log(1+exp(-r*y))
            self.loss_derivative = lambda r, y: -(y/(1+exp(r*y)))
            self.task='classification'
        else:
            raise Exception('Not implemented loss "%s"'%loss)
    
    
    def fit(self, X, y, X_val=None, y_val=None, bad_iters_count=+inf):
        '''If called like fit(X,y), then the number of base_learners is always base_learners_count (specified at initialization).
        If called like fit(self, X, y, X_val, y_val, bad_iters_count) at most there are also base_learners_count base learners but may be less due to 
        early stopping:
        At each iteration accuracy (using validation set, specified by X_val [design matrix], y_val [outputs]) is estimated and 
        position of best iteration tracked. If there were >=bad_iters_count after the best iteration, fitting process stops.'''
        
        X=X.astype(float32)
        y=y.astype(float32)
        D=X.shape[1]
        min_loss = +inf
        min_pos = -1
        
        if self.task=='classification':
            assert all(unique(y)==[0,1]),'Only y=0 or y=1 supported!'
            y[y==0]=-1 # inner format of classes y=+1 or y=-1
        N=len(X)
        F_current = self.F(X) # current value, all zeros if not tuned before.
        if X_val!=None and y_val!=None:
            F_val = self.F(X_val)
        
        for iter_num in piter(range(self.base_learners_count), percent_period=3, show=(self.log.level>=2)):
            
            if X_val!=None and y_val!=None:
                if self.task=='regression':
                    Y_val_hat = F_val
                    loss = mean(abs(Y_val_hat-y_val))  # MAE tracking on validation set
                else:  # classification
                    Y_val_hat = (F_val>=0).astype(int)    
                    loss = 1-skl.metrics.accuracy_score(y_val,Y_val_hat)            
                if loss<min_loss:
                    min_pos = iter_num
                    min_loss = loss
                    
                if iter_num-min_pos>=bad_iters_count:
                    self.base_learners_count = len(self.base_learners)
                    self.log.pr1('\nEarly stopping with %d base lerners, because last %d losses were above min_loss=%.3f at position %d.' % (self.base_learners_count, 
                                                                                                                                    bad_iters_count,
                                                                                                                                    min_loss,
                                                                                                                                    min_pos))
                    break
            
            z = -self.loss_derivative(F_current, y)
            
            base_learner = self.base_learner.__class__(**self.base_learner.get_params())   # recreate base learner
            base_learner.fit(X,z)
            
            if isinstance(base_learner,skl.tree.tree.DecisionTreeRegressor) and (self.refit_tree==True): # tree refitting
                leaf_ids = base_learner.tree_.apply(X)
                unique_leaf_ids = unique(leaf_ids)
                for leaf_id in unique_leaf_ids:
                    leaf_pos_sels = (leaf_ids==leaf_id)
                    prediction = base_learner.tree_.value[leaf_id,0,0]
                   
                    def loss_at_leaf(value):
                        return np.sum(self.loss(F_current[leaf_pos_sels]+value, y[leaf_pos_sels]))
                    
                    refined_prediction = scipy.optimize.fmin(loss_at_leaf, prediction,
                                                             xtol=self.optimization.xtol, ftol=self.optimization.ftol, 
                                                             maxfun=self.optimization.max_fun_evals, disp=0)
                    
                    base_learner.tree_.value[leaf_id,0,0] = refined_prediction

            base_pred = base_learner.predict(X)
            
            if self.fit_coefs==False:  # coefficients by base learner refitting
                coef = 1
            else:
                def loss_after_weighted_addition(coef):
                    return np.sum(self.loss(F_current+coef*base_pred, y))
                
                res = scipy.optimize.fmin(loss_after_weighted_addition, 1, 
                                          xtol=self.optimization.xtol, ftol=self.optimization.ftol,  
                                          maxfun=self.optimization.max_fun_evals, disp=0)
                coef = res[0]
                if coef<0:
                    self.log.pr3('coef=%s is negative!' % coef)
                if coef==0:
                    self.log.pr3('coef=%s is zero!' % coef)
                
            coef = coef*self.shrinkage
            
            self.coefs.append(coef)       
            self.base_learners.append(base_learner) 
            
            F_current += coef*base_pred            
            if X_val!=None and y_val!=None:
                F_val += coef*base_learner.predict(X_val) 

                    
                   
    def F(self, X, max_base_learners_count=inf):
        '''Internal function used for forecasting. 
        X-design matrix, each row is an object for which a forecast should be made.
        max_base_learners_count - maximal iteration at which to stop. F is evaluated for min(max_base_learners_count, len(self.base_learners)) models.'''
        
        F_val = zeros(len(X))

        for iter_num, (coef, base_learner) in enumerate(zip(self.coefs, self.base_learners)):
            base_pred = base_learner.predict(X)
            F_val += coef*base_pred
            if iter_num+1>=max_base_learners_count:
                break            
    
        return F_val
    
    
    def predict(self, X, base_learners_count=inf):
        if self.task=='regression':
            return self.F(X,base_learners_count)
        else:  # classification
            return (self.F(X,base_learners_count)>=0).astype(int)   # F(X)>=0 = > predition=1 otherwise prediction=0
    
    
    def predict_proba(self, X, base_learners_count=inf):
        '''Predict class probabilities for objects, specified by rows of matrix X. 
        iter_num - at what iteration to stop. If not specified all base learners are used.
        Applicable only for loss function="log". Classes are stored in self.classes_ attribute.'''
        
        if self.loss!='log':
            raise Exception('Inapliccable for loss %s'%self.loss)
        self.classes_ = [0, 1]
        scores = self.F(X, base_learners_count)
        probs = 1/(1+exp(-scores))
        return hstack([1-probs,probs])
    

    def get_losses(self,X,Y):
        '''Estimate loss for each iteration of the prediction process of boosting.
        Returns an array losses with length=#{base learners}'''
        
        losses = zeros(len(self.base_learners))
        F_val = zeros(len(X))

        for iter_num, (coef, base_learner) in enumerate(zip(self.coefs, self.base_learners)):
            base_pred = base_learner.predict(X)
            F_val += coef*base_pred

            if self.task=='regression':
                Y_hat = F_val
                losses[iter_num] = (mean(abs(Y_hat-Y))/mean(abs(Y)))
            else:  # classification
                Y_hat = (F_val>=0).astype(int)    
                losses[iter_num] = 1-skl.metrics.accuracy_score(Y,Y_hat)

        return losses
    
    
    
    def remove_redundant_base_learners(self, X_val, Y_val, max_count=+inf):
        '''Using validation set, specified by (X_val, y_val) find optimal number of base learners 
        (at this number the loss on validation is minimal).
        All base learners above this number are removed. 
        max_count - is the maximum possible number of base learners retained.'''
        
        orig_count = len(self.base_learners)
        losses = self.get_losses(X_val, Y_val)
        self.base_learners_count = min(argmin(losses)+1, max_count)
        self.coefs = self.coefs[:self.base_learners_count]
        self.base_learners = self.base_learners[:self.base_learners_count]
        self.log.pr1('Cut base learners count from %d to %d.' %(orig_count, self.base_learners_count))