import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmultiflow.trees import HoeffdingTree
from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import KNN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection import KSWIN
from skmultiflow.drift_detection.hddm_w import HDDM_W
import json
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from skmultiflow.data import DataStream
from skmultiflow.meta import AdditiveExpertEnsemble
from scipy.stats import bernoulli
#from sklearn.svm import SVC
from skmultiflow.neural_networks import PerceptronMask
import sys
import time
from my_additive_expert_ensemble import MyAdditiveExpertEnsemble
import seaborn as sns
from skmultiflow.meta import OzaBaggingADWINClassifier
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import AdditiveExpertEnsembleClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import OnlineSMOTEBaggingClassifier
from skmultiflow.meta import OnlineAdaC2Classifier
from skmultiflow.data import AGRAWALGenerator

#-----------------------------------------------------------------------------------------------------------------------------------------------
#def compute_bounds(p_hat, decay, n, alpha, n_sim=1000):
    #bernoulli_samples = bernoulli.rvs(p_hat, size=n * n_sim).reshape(n_sim, n)
    # TODO: Check if shapes match
    ##empirical_bounds = (1 - decay) * (bernoulli_samples * (n - np.arange(1, n + 1))).sum(axis=1)
    #empirical_bounds = (1 - decay) * np.matmul(bernoulli_samples, decay ** (n - np.arange(1, n + 1)).reshape(n, 1)).sum(axis=1)
    #lb, ub = np.percentile(empirical_bounds, q=[alpha * 100, (1 - alpha) * 100])
    #return lb, ub
#def find_nearest(array, value):
    #idx = np.searchsorted(array, value, side='left')
    #if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        #return array[idx-1]
    #else:
        #return array[idx]

"""
class BoundsTable(object):
    def __init__(self, decay, alpha_range, p_hat_range, n_range, n_sim=1000):
        self.decay = decay
        self.alpha_range = alpha_range
        self.p_hat_range = p_hat_range
        self.n_range = n_range
        self.n_sim = n_sim
        self.bounds_table = {}
    def compute_bounds_table(self, rng_seed=123321):
        np.random.seed(rng_seed)
        grid = itertools.product(self.p_hat_range, self.n_range, self.alpha_range)
        # TODO: Is it safe to store dictionary keys as floating point values?
        self.bounds_table = {(p, n, alpha): compute_bounds(p_hat=p, alpha=alpha, decay=self.decay, n=n, n_sim=self.n_sim)
                             for (p, n, alpha) in grid}
        return self
    def lookup_bounds(self, p, n, alpha):
        # We assume here that n and alpha can be exactly matched
        p_nearest = find_nearest(self.p_hat_range, p)
        return self.bounds_table[(p_nearest, n, alpha)]
#---------------------------------------------------------------------------------------------------------------------------------------------------
"""
class Potential_drift_detection:
	def __init__(self,T_pot=0):
		self.count=0
		self.tpr=0
		self.tnr=1
		self.ppv=2
		self.npv=3
		self.R = np.zeros([1500, 4], dtype = float)
		self.P = np.zeros([1500, 4], dtype = float)
		self.astrick=[self.tpr,self.tnr,self.ppv,self.npv]
		self.eta = np.empty(4, dtype = float)
		self.delta = np.empty(4, dtype = float)
		self.epsilon = np.empty(4, dtype = float)
		self.T_pot=T_pot
		self.C=np.zeros([1500,2,2], dtype=int)
		self.C_total=np.zeros([1500,2,2], dtype=int)
		#self.create_stream()
		self.y_cap=np.zeros(1500,dtype=int)
		self.y=np.zeros(1500,dtype=int)
		self.eta[self.tpr]=.9
		self.eta[self.tnr]=.9
		self.eta[self.ppv]=.9
		self.eta[self.npv]=.9
		self.iteration=300
		self.train_on=2000
		self.no_of_drifts=0
	#------------------------------------------------------------------------------------------------------------------------------------------
	#def _compute_bounds_table(self, n_samples, rng_seed=123321):
        #alpha_range = np.array([self.warn_level, self.detect_level])
        #p_hat_range = np.arange(1, 100) / 100.0
        #n_range = np.arange(2, n_samples + 1)
        #self.bounds_table = BoundsTable(self.decay, alpha_range, p_hat_range, n_range, n_sim=self.n_sim)
        #self.bounds_table.compute_bounds_table(rng_seed=rng_seed)
		#return self.bounds_table
	#-------------------------------------------------------------------------------------------------------------------------------------------
	def bound_table(self,P,eta,delta,n):
		alpha=delta
		sum=0
		I=np.zeros(n+1,dtype=int)
		R=np.zeros(500)
		x=[y for  y in range(0, 500)]
		for j in range(0,500):
			sum=0
			for i in range(1,n+1):
				I[i]=bernoulli.rvs(size=1,p=P)
				#print("I:",i,":",I[i])
				#print("eta",eta,"n=",n,"i=",i)
				a=(eta**(n-i))*I[i]
				#print("a:",a)
				sum=sum+(eta**(n-i))*I[i]
				#print("sum=",sum)
			#print("All I printed-=-=-=-=-=-=-=-=-=-=-=--=-=-=--=")
			R[j]=(1-eta)*sum

			#print("--j----",j,"-R[j]-",R[j])
		'''sns.set_style("whitegrid")
		plt.xlabel("R[j]")
		plt.ylabel("j")
		plt.title('R[j] distribution')
		#plt.axis([0,350,0,1.0])
		plt.plot(x, R)
		#plt.scatter(self.drift,list_iter)
		plt.draw()
		plt.show()'''
		lb=np.quantile(R,alpha,axis=None, out=None, overwrite_input=False, interpolation='higher', keepdims=False)
		#print("Lower Bound:","---",lb)
		ub=np.quantile(R,1-alpha,axis=None, out=None, overwrite_input=False, interpolation='higher', keepdims=False)
		#print("Upper Bound:","---",ub)
		return lb,ub
		#plt.plot(range(0,100),R)
		#plt.show()	
		

	def four_linear_rates_calculation(self):
		temp=['tpr','tnr','ppv','npv']
		warn_time=0
		detect_time=0
		N=np.zeros(4,dtype=int)
		warn_bd_lb=np.zeros(4,dtype=float)
		warn_bd_ub=np.zeros(4,dtype=float)
		detect_bd_lb=np.zeros(4,dtype=float)
		detect_bd_ub=np.zeros(4,dtype=float)
		self.drift=[]
		#CL=HoeffdingTree()
		#nn=PerceptronMask(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, n_iter=None)
		CL=NaiveBayes()
		#CL = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
		#ht.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		#clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',  random_state=None)
		CL.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		self.aee=MyAdditiveExpertEnsemble(n_estimators=20, base_estimator=CL, beta=0.4, gamma=0.6, pruning='weakest')
		self.aee.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		print('creation of additive expert ensemble (aee) done with NaiveBayes as base classifier')
		for x in range(0,4):
			self.R[0][x]=0.0
			self.P[0][x]=0.0
		print("initializing C")
		for x in range(0,2):
			for n in range(0,2):
				self.C[0][x][n]=1
				self.C_total[x][n]=1
				#print(x,",",n,":",self.C[0][x][n])
		for t in range(1,self.iteration):
			print("iteration no: ",t)
			sample=self.ds.next_sample()
			self.y[t]=sample[1]
			self.y_cap[t]=self.aee.predict(sample[0])
			for x in range(0,2):
				for n in range(0,2):
					self.C[t][x][n]=self.C[t-1][x][n]
					self.C_total[t][x][n]=self.C_total[t-1][x][n]
			print("y_cap=",self.y_cap[t],'y=',self.y[t])
			self.C[t][self.y_cap[t]][self.y[t]]=self.C[t-1][self.y_cap[t]][self.y[t]]+1	
			self.C_total[t][self.y_cap[t]][self.y[t]]=self.C_total[t-1][self.y_cap[t]][self.y[t]]+1
			'''print('printing C')
			for x in range(0,2):
				for y in range(0,2):
					print(x,",",y,":",self.C[t][x][y])'''
			for x in self.astrick:
				
				if((self.y[t]==1)and (x==self.tpr)):
					if(self.y[t]==self.y_cap[t]):
						self.R[t][x]=self.eta[x]*self.R[t-1][x]+(1-self.eta[x])
					else:
						self.R[t][x]=self.eta[x]*self.R[t-1][x]
				elif((self.y[t]==0)and (x==self.tnr)):
					if(self.y[t]==self.y_cap[t]):
						self.R[t][x]=self.eta[x]*self.R[t-1][x]+(1-self.eta[x])
					else:
						self.R[t][x]=self.eta[x]*self.R[t-1][x]
				elif((self.y_cap[t]==1)and (x==self.ppv)):
					if(self.y[t]==self.y_cap[t]):
						self.R[t][x]=self.eta[x]*self.R[t-1][x]+(1-self.eta[x])
					else:
						self.R[t][x]=self.eta[x]*self.R[t-1][x]
				elif((self.y_cap[t]==0)and (x==self.npv)):
					if(self.y[t]==self.y_cap[t]):
						self.R[t][x]=self.eta[x]*self.R[t-1][x]+(1-self.eta[x])
					else:
						self.R[t][x]=self.eta[x]*self.R[t-1][x]
				else:
					self.R[t][x]=self.R[t-1][x]
					
			#printing R
			for x in self.astrick:
				#print("For ",temp[x],":")
				#print("R:",temp[x],"::",self.R[t][x])
					
				if x in [self.tpr, self.tnr]:
					if x==self.tpr:
						N[x]=self.C[t][0][1]+self.C[t][1][1]
						#print("N:",temp[x],"::",N[x])
						self.P[t][x]=(self.C[t][1][1])/N[x]
						#print("P:",t,temp[x],"::",self.P[t][x])
					else:
						N[x]=self.C[t][0][0]+self.C[t][1][0]
						#print("N:",temp[x],"::",N[x])
						self.P[t][x]=(self.C[t][0][0])/N[x]
						#print("P:",t,temp[x],"::",self.P[t][x])
				else:
					if x==self.ppv:
						N[x]=self.C[t][1][0]+self.C[t][1][1]  
						#print("N:",temp[x],"::",N[x])
						self.P[t][x]=self.C[t][1][1]/N[x]
						#print("P:",t,temp[x],"::",self.P[t][x])
					else:
						N[x]=self.C[t][0][0]+self.C[t][0][1]
						#print("N:",temp[x],"::",N[x])
						self.P[t][x]=self.C[t][0][0]/N[x]
						#print("P:",t,temp[x],"::",self.P[t][x])
				#print("P:",temp[x],"::",self.P[t][x])
				temp2=self.P[t][x]
				#print("for ",x)
				#print("P:",self.P[t][x],"eta:",self.eta[x],"N[x]",N[x])
				#print("for Warning:")
				warn_bd_lb[x],warn_bd_ub[x]=self.bound_table(temp2,self.eta[x],.3,N[x])
				#print("Warn_bd:",temp[x],"::",warn_bd[x])
				#print("for detection:")
				detect_bd_lb[x],detect_bd_ub[x]=self.bound_table(temp2,self.eta[x],.01,N[x])
				#print("detect_bd:",temp[x],"::",detect_bd[x])
				
				#------------------------------------------------------------------------------------------------------------------
				#lb_warn, ub_warn = self.bounds_table.lookup_bounds(p=p_hat, n=n, alpha=self.warn_level)
                #lb_detect, ub_detect = self.bounds_table.lookup_bounds(p=p_hat, n=n, alpha=self.detect_level)
				#------------------------------------------------------------------------------------------------------------------
			#if((self.R[t][x]>warn_bd[x] for x in self.astrick) and warn_time==0):
			#	warn_time=t
			#elif( self.R[t][x]<= warn_bd[x] for x in self.astrick):
			#	warn_time=0
			
			#for x in self.astrick:
			#	if((self.R[t][x]>warn_bd[x]) and warn_time==0):
			#		warn_time=t
			#		break
			#	elif(self.R[t][x]<=warn_bd[x]):
			#		warn_time=0
			
			temp3=0
			for x in self.astrick:
				if(self.R[t][x]>warn_bd_ub[x]): #warn_bd_lb[x]>self.R[t][x]) or
					print('at warning level: R:',self.R[t][x],'warn_bd_lb:',warn_bd_lb[x], 'warn_bd_ub:',warn_bd_ub[x])
					temp3=1
					if(warn_time==0):
						warn_time=t
						break;
						
			if temp3==0:
				warn_time=0
			
			#for x in self.astrick:
				#print("check for detection: R:", self.R[t][x], "detect_bd_lb:", detect_bd_lb[x]," detect_bd_ub: ", detect_bd_ub[x])
			for x in self.astrick:
				if((self.R[t][x]>detect_bd_ub[x])):   #(detect_bd_lb[x]>self.R[t][x])or
					#print('at detect level: R',self.R[t][x],'detect_bd_lb',detect_bd_lb[x], 'detect_bd_ub',detect_bd_ub[x])
					self.count=self.count+1
					detect_time=t
					print("Warn Time:",warn_time,"   Detect Time:",detect_time)
					self.drift.append(detect_time)
					new_warn_time=warn_time
					if(detect_time-warn_time)<50:
						add_data=detect_time-50
						if add_data<0:
							new_warn_time=0
						else:
							new_warn_time=add_data
					#print("new_warn_time: ",new_warn_time)	
					X_retrain_data=self.a[new_warn_time:detect_time]
					y_retrain_data=self.tar[new_warn_time:detect_time]
					new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.6)
					new_exp.estimator.fit(X_retrain_data,y_retrain_data)
					#self.aee._add_expert( new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.1))
					#self.aee.partial_fit(X_retrain_data,y_retrain_data)
					warn_time=0
					self.aee._add_expert( new_exp)
					print("Classifier's Info: ", self.aee.get_params(deep=True))
					#reset R, P, C
					for x in range(0,4):
						self.R[t][x]=0.5
						self.P[t][x]=0.5
					for x in range(0,2):
						for n in range(0,2):
							self.C[t][x][n]=1
					break
			print("Count:",self.count)
		self.no_of_drifts=self.count
				
				
				
				
				
				
				
	
	#creating stream of text data
	def create_stream(self):
		#Reading reviews from json file
		reviews = []
		for line in open('Cell_Phones_and_Accessories_5.json', 'r'):
			reviews.append(json.loads(line))
		
		#extracting reviewText and review_star 
		review_text=[]
		review_star=[]
		
		for l in range(0,len(reviews)):
			review_text.append(reviews[l]['reviewText'])
			review_star.append(reviews[l]['overall'])
		
		#creating dataset with 23000 reviews
		review_set=[]
		review_star_set=[]
		c1=c2=c3=c4=c5=0
		i=5
		for l in range(0,len(reviews)):
			if(reviews[l]['overall']==5 and c5<5000 and i>2):
				review_set.append(reviews[l]['reviewText'])
				review_star_set.append(reviews[l]['overall'])
				c5=c5+1
				i=i-1
			if(reviews[l]['overall']==4 and c4<5000 and i>2):
				review_set.append(reviews[l]['reviewText'])
				review_star_set.append(reviews[l]['overall'])
				c4=c4+1
				i=2
			if(reviews[l]['overall']==3 and c3<3000 and i<=2):
				review_set.append(reviews[l]['reviewText'])
				review_star_set.append(reviews[l]['overall'])
				c3=c3+1
				i=i+1
			if(reviews[l]['overall']==2 and c2<5000 and i<=2):
				review_set.append(reviews[l]['reviewText'])
				review_star_set.append(reviews[l]['overall'])
				c2=c2+1
				i=i+1
			if(reviews[l]['overall']==1 and c1<5000 and i<=2):
				review_set.append(reviews[l]['reviewText'])
				review_star_set.append(reviews[l]['overall'])
				c1=c1+1
				i=5
		

		#labeling reviews on the bases of given star rating
		review_sentiment=[]
		#len(review_sentiment)
		p=0
		for l in range(0,len(review_star_set)):
   
			if review_star_set[l]==4.0 or review_star_set[l]==5.0:
        
				p=1
			else: 
				if(review_star_set[l]==3.0):
					p=0
				else:
					if(review_star_set[l]==1.0 or review_star_set[l]==2.0):
						p=0
			review_sentiment.append(p)
		
		
		self.review_text_part1=review_set[0:self.train_on]
		self.review_text_part2=review_set[self.train_on:]
		self.review_sentiment_part1=review_sentiment[0:self.train_on]
		self.review_sentiment_part2=review_sentiment[self.train_on:]
		count_1=0
		count_0=0
		for x in range(self.train_on):
			if(self.review_sentiment_part1[x]==0):
				count_0=count_0+1
			else:
				count_1=count_1+1
		print("In training:: 0:",count_0,"  1:",count_1)
		count_0=0
		count_1=0
		for x in range(self.iteration):
			if(self.review_sentiment_part2[x]==0):
				count_0=count_0+1
			else:
				count_1=count_1+1
		print("In Testing:: 0:",count_0,"  1:",count_1)
				
		
		#Creating document term matrix of the extracted review text
		vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_features=500)
		vect.fit(review_set)
		self.X_review_dtm1=vect.transform(self.review_text_part1)
		self.X_review_dtm2=vect.transform(self.review_text_part2)
		
		#-----------????????------------
		#target=pd.DataFrame(review_sentiment)
		#X_df=pd.DataFrame(X_review_dtm)
		
		#Converting sparse matrix into dense array for use in datastream creation
		self.a=self.X_review_dtm2.toarray()
		self.tar=np.array(self.review_sentiment_part2)
		
		#creaing datastream
		self.ds=DataStream(self.a,y=self.tar)
		# preparing datastream for use
		self.ds.prepare_for_use()
		
		#testing datastream
		#print(self.ds.n_features)
	
	
	def create_stream_amazon(self):
		
		location=r'sentiment labelled sentences\amazon_cells_labelled.txt'
		df=pd.read_csv(location,sep='\t',names=('Sentence','label'))
		simple_Train=df['Sentence'].tolist()
		Test=df['label'].tolist()
		print(type(simple_Train))
		print(type(Test))
		
		Train_data=simple_Train[0:300]
		Test_data=simple_Train[300:]
		Train_label=Test[0:300]
		Test_label=Test[300:]
		self.review_sentiment_part1=Test[0:300]
		
		#Creating document term matrix of the extracted review text
		vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_features=500)
		vect.fit(Train_data)
		self.X_review_dtm1=vect.transform(Train_data)
		self.X_review_dtm2=vect.transform(Test_data)
		
		#Converting sparse matrix into dense array for use in datastream creation
		self.a=self.X_review_dtm2.toarray()
		self.tar=np.array(Test_label)
		
		#creaing datastream
		self.ds=DataStream(self.a,y=self.tar)
		# preparing datastream for use
		self.ds.prepare_for_use()
		print(len(Train_data))
		print(len(Test_data))
		print(len(Train_label))
		print(len(Test_label))
		
		print(df.head())
		
	def create_stream_imdb(self):
		
		location=r'sentiment labelled sentences\imdb_labelled.txt'
		df=pd.read_csv(location,sep='\t',names=('Sentence','label'))
		#print(df[855:860])
		simple_Train=df['Sentence'].tolist()
		Test=df['label'].tolist()
		print(len(simple_Train))
		print(len(Test))
		
		Train_data=simple_Train[0:700]
		Test_data=simple_Train[700:]
		Train_label=Test[0:700]
		Test_label=Test[700:]
		self.review_sentiment_part1=Test[0:700]
		
		#Creating document term matrix of the extracted review text
		vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_features=500)
		vect.fit(Train_data)
		self.X_review_dtm1=vect.transform(Train_data)
		self.X_review_dtm2=vect.transform(Test_data)
		
		#Converting sparse matrix into dense array for use in datastream creation
		self.a=self.X_review_dtm2.toarray()
		self.tar=np.array(Test_label)
		
		#creaing datastream
		self.ds=DataStream(self.a,y=self.tar)
		# preparing datastream for use
		self.ds.prepare_for_use()
		print(len(Train_data))
		print(len(Test_data))
		print(len(Train_label))
		print(len(Test_label))
	
	def create_stream_yelp(self):
		
		location=r'sentiment labelled sentences\yelp_labelled.txt'
		df=pd.read_csv(location,sep='\t',names=('Sentence','label'))
		#print(df[855:860])
		simple_Train=df['Sentence'].tolist()
		Test=df['label'].tolist()
		print(len(simple_Train))
		print(len(Test))
		
		Train_data=simple_Train[0:700]
		Test_data=simple_Train[700:]
		Train_label=Test[0:700]
		Test_label=Test[700:]
		self.review_sentiment_part1=Test[0:700]
		
		#Creating document term matrix of the extracted review text
		vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_features=500)
		vect.fit(Train_data)
		self.X_review_dtm1=vect.transform(Train_data)
		self.X_review_dtm2=vect.transform(Test_data)
		
		#Converting sparse matrix into dense array for use in datastream creation
		self.a=self.X_review_dtm2.toarray()
		self.tar=np.array(Test_label)
		
		#creaing datastream
		self.ds=DataStream(self.a,y=self.tar)
		# preparing datastream for use
		self.ds.prepare_for_use()
		print(len(Train_data))
		print(len(Test_data))
		print(len(Train_label))
		print(len(Test_label))
		
		print(df.head())	
	def create_stream_mixed(self):
		
		location=r'sentiment labelled sentences\My_data.txt'
		df=pd.read_csv(location,sep='\t',names=('Sentence','label'))
		simple_Train=df['Sentence'].tolist()
		Test=df['label'].tolist()
		
		print(len(simple_Train))
		print(len(Test))
		
		Train_data=simple_Train[0:600]
		Test_data=simple_Train[600:]
		Train_label=Test[0:600]
		Test_label=Test[600:]
		self.review_sentiment_part1=Test[0:600]
		
		#Creating document term matrix of the extracted review text
		vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_features=500)
		vect.fit(Train_data)
		self.X_review_dtm1=vect.transform(Train_data)
		self.X_review_dtm2=vect.transform(Test_data)
		
		#Converting sparse matrix into dense array for use in datastream creation
		self.a=self.X_review_dtm2.toarray()
		self.tar=np.array(Test_label)
		
		#creaing datastream
		self.ds=DataStream(self.a,y=self.tar)
		# preparing datastream for use
		self.ds.prepare_for_use()
		print(len(Train_data))
		print(len(Test_data))
		print(len(Train_label))
		print(len(Test_label))
		
		print(df.head())	
	
	def DDM_detection(self):
		warn_time=0
		detect_time=0
		drift_count=0
		correct=0
		incorrect=0
		CL = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
		#CL=NaiveBayes()
		#CL=HoeffdingTree()
		CL.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		self.aee=MyAdditiveExpertEnsemble(n_estimators=20, base_estimator=CL, beta=0.5, gamma=0.2, pruning='weakest')
		self.aee.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		ddm = DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)
		for i in range(self.iteration):
			print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=self.aee.predict(sample[0])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
			ddm.add_element(pred)
			if ddm.detected_warning_zone() and warn_time==0:
				print('Warning zone has been detected in data:  - of index: ' + str(i))
				warn_time=i
			
			if ddm.detected_change():
				drift_count=drift_count+1
				print('Change has been detected in data:  - of index: ' + str(i))
				detect_time=i
				print("Warn Time:",warn_time,"   Detect Time:",detect_time)
				X_retrain_data=self.a[warn_time:detect_time]
				y_retrain_data=self.tar[warn_time:detect_time]
				self.aee._add_expert( new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.2))
				self.aee.partial_fit(X_retrain_data,y_retrain_data)
				print("Classifier's Info: ", self.aee.get_params(deep=True))
				warn_time=0
		accuracy=(correct/(correct+incorrect))*100;
		print("Accuracy:",accuracy)
		#self.no_of_drifts=drift_count
		print("Total number of Drift: ",drift_count)
			
	def EDDM_detection(self):
		warn_time=0
		detect_time=0
		drift_count=0
		correct=0
		incorrect=0
		CL = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
		#CL=NaiveBayes()
		#CL=HoeffdingTree()
		#nb.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		self.aee=MyAdditiveExpertEnsemble(n_estimators=2, base_estimator=CL, beta=0.5, gamma=0.2, pruning='weakest')
		self.aee.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1 ,  classes=[0,1], sample_weight=None)
		eddm = EDDM()
		for i in range(self.iteration):
			print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=self.aee.predict(sample[0])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
			eddm.add_element(pred)
			if eddm.detected_warning_zone() and warn_time==0:
				print('Warning zone has been detected in data:  - of index: ' + str(i))
				warn_time=i
			
			if eddm.detected_change():
				drift_count=drift_count+1
				print('Change has been detected in data:  - of index: ' + str(i))
				detect_time=i
				print("Warn Time:",warn_time,"   Detect Time:",detect_time)
				X_retrain_data=self.a[warn_time:detect_time]
				y_retrain_data=self.tar[warn_time:detect_time]
				self.aee._add_expert( new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.2))
				self.aee.partial_fit(X_retrain_data,y_retrain_data)
				print("Classifier's Info: ", self.aee.get_params(deep=True))
				warn_time=0
		accuracy=(correct/(correct+incorrect))*100;
		print("Accuracy:",accuracy)
		#self.no_of_drifts=drift_count
		print("Total number of Drift: ",drift_count)
	#------------------------------------------------------KSWIN DRIFT DETECTION METHOD-------------------------------------------------------------------------------#	
	def KSWIN_detection(self):
		warn_time=0
		detect_time=0
		drift_count=0
		correct=0
		incorrect=0
		#CL = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
		#CL=NaiveBayes()
		CL=HoeffdingTree()
		#nb.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		self.aee=MyAdditiveExpertEnsemble(n_estimators=2, base_estimator=CL, beta=0.5, gamma=0.2, pruning='weakest')
		self.aee.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1 ,  classes=[0,1], sample_weight=None)
		kswin = KSWIN(alpha=0.005)
		print( " KSWIN Drift Detection Method is implemented ")
		for i in range(self.iteration):
			print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=self.aee.predict(sample[0])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
			kswin.add_element(pred)
			if kswin.detected_warning_zone() and warn_time==0:
				print('Warning zone has been detected in data:  - of index: ' + str(i))
				warn_time=i
			
			if kswin.detected_change():
				drift_count=drift_count+1
				print('Change has been detected in data:  - of index: ' + str(i))
				detect_time=i
				print("Warn Time:",warn_time,"   Detect Time:",detect_time)
				X_retrain_data=self.a[warn_time:detect_time]
				y_retrain_data=self.tar[warn_time:detect_time]
				self.aee._add_expert( new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.2))
				self.aee.partial_fit(X_retrain_data,y_retrain_data)
				print("Classifier's Info: ", self.aee.get_params(deep=True))
				warn_time=0
		accuracy=(correct/(correct+incorrect))*100;
		print("Accuracy:",accuracy)
		#self.no_of_drifts=drift_count
		print("Total number of Drift: ",drift_count)
	#----------------------------------------------------------------------------------------------------------------------------------------------------------#
	
	
	#------------------------------------------------------HDDM_W DRIFT DETECTION METHOD-------------------------------------------------------------------------------#	
	def HDDM_W_detection(self):
		warn_time=0
		detect_time=0
		drift_count=0
		correct=0
		incorrect=0
		#CL = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
		CL=NaiveBayes()
		#CL=HoeffdingTree()
		#nb.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		self.aee=MyAdditiveExpertEnsemble(n_estimators=2, base_estimator=CL, beta=0.5, gamma=0.2, pruning='weakest')
		self.aee.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1 ,  classes=[0,1], sample_weight=None)
		hddm_w = HDDM_W(drift_confidence=0.005, warning_confidence=0.01, lambda_option=0.05, two_side_option=True)
		print( " KSWIN Drift Detection Method is implemented ")
		for i in range(self.iteration):
			print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=self.aee.predict(sample[0])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
			hddm_w.add_element(pred)
			if hddm_w.detected_warning_zone() and warn_time==0:
				print('Warning zone has been detected in data:  - of index: ' + str(i))
				warn_time=i
			
			if hddm_w.detected_change():
				drift_count=drift_count+1
				print('Change has been detected in data:  - of index: ' + str(i))
				detect_time=i
				print("Warn Time:",warn_time,"   Detect Time:",detect_time)
				X_retrain_data=self.a[warn_time:detect_time]
				y_retrain_data=self.tar[warn_time:detect_time]
				self.aee._add_expert( new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.2))
				self.aee.partial_fit(X_retrain_data,y_retrain_data)
				print("Classifier's Info: ", self.aee.get_params(deep=True))
				warn_time=0
		accuracy=(correct/(correct+incorrect))*100;
		print("Accuracy:",accuracy)
		#self.no_of_drifts=drift_count
		print("Total number of Drift: ",drift_count)
	#----------------------------------------------------------------------------------------------------------------------------------------------------------#
	#------------------------------------------------------HDDM_W---- with single classifier-------------------------------------------------------------------#
	def HDDM_W_SINGLE_CLASSIFIER_detection(self):
		warn_time=0
		detect_time=0
		drift_count=0
		correct=0
		incorrect=0
		CL = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
		#CL=NaiveBayes()
		#CL=HoeffdingTree()
		#nb.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		#self.aee=MyAdditiveExpertEnsemble(n_estimators=2, base_estimator=CL, beta=0.5, gamma=0.2, pruning='weakest')
		CL.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1 ,  classes=[0,1], sample_weight=None)
		hddm_w = HDDM_W(drift_confidence=0.005, warning_confidence=0.01, lambda_option=0.05, two_side_option=True)
		print( "  HDDM_W Drift Detection Method is implemented ")
		for i in range(self.iteration):
			print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=CL.predict(sample[0])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
			hddm_w.add_element(pred)
			if hddm_w.detected_warning_zone() and warn_time==0:
				print('Warning zone has been detected in data:  - of index: ' + str(i))
				warn_time=i
			
			if hddm_w.detected_change():
				drift_count=drift_count+1
				print('Change has been detected in data:  - of index: ' + str(i))
				detect_time=i
				print("Warn Time:",warn_time,"   Detect Time:",detect_time)
				X_retrain_data=self.a[warn_time:detect_time]
				y_retrain_data=self.tar[warn_time:detect_time]
				#self.aee._add_expert( new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.2))
				CL.partial_fit(X_retrain_data,y_retrain_data)
				#print("Classifier's Info: ", self.aee.get_params(deep=True))
				warn_time=0
		accuracy=(correct/(correct+incorrect))*100;
		print("Accuracy:",accuracy)
		#self.no_of_drifts=drift_count
		print("Total number of Drift: ",drift_count)
	
	def page_hinkley_detection(self):
		warn_time=0
		detect_time=0
		drift_count=0
		correct=0
		incorrect=0
		nb=NaiveBayes()
		nb.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1)
		self.aee=MyAdditiveExpertEnsemble(n_estimators=2,base_estimator=nb, beta=0.4, gamma=0.6, pruning='oldest')
		self.aee.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		ph = PageHinkley()
		for i in range(self.iteration):
			print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=self.aee.predict(sample[0])
			print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
			ph.add_element(pred)
			'''
			if ddm.detected_warning_zone() and warn_time==0:
				print('Warning zone has been detected in data:  - of index: ' + str(i))
				warn_time=i
			'''
			if ph.detected_change():
				drift_count=drift_count+1
				print('Change has been detected in data:  - of index: ' + str(i))
				detect_time=i
				warn_time=detect_time-30
				print("Warn Time:",warn_time,"   Detect Time:",detect_time)
				X_retrain_data=self.a[warn_time:detect_time]
				y_retrain_data=self.tar[warn_time:detect_time]
				self.aee._add_expert( new_exp=self.aee._construct_new_expert(weight= self.aee.get_ensemble_weight()*0.2))
				self.aee.partial_fit(X_retrain_data,y_retrain_data)
				print("Classifier's Info: ", self.aee.get_params(deep=True))
				warn_time=0
		accuracy=(correct/(correct+incorrect))*100;
		print("Accuracy:",accuracy)
		#self.no_of_drifts=drift_count
		print("Total number of Drift: ",drift_count)
	
	
	def print_prediction(self):
		correct=0
		incorrect=0
		acc_list=np.empty(self.iteration)
		print("Printing Confusion Matrix: ")
		for x in range(0,2):
				for y in range(0,2):
					print(x,",",y,":",self.C_total[self.iteration-1][x][y])
		for x in range(1,self.iteration):
			if(self.y[x]==self.y_cap[x]):
				correct=correct+1
			else:
				incorrect=incorrect+1
				
		accuracy=(correct/(correct+incorrect))*100;
		print("Accuracy:",accuracy)
		print("Total no of drifts: ",self.no_of_drifts)
		list_iter=np.arange(0,self.iteration,1)
		for x in range(0,self.iteration):
			tn=self.C_total[x][0][0]
			tp=self.C_total[x][1][1]
			fn=self.C_total[x][0][1]
			fp=self.C_total[x][1][0]
			acc_list[x]= ((tp+tn)/(tp+tn+fp+fn))
			
		sns.set_style("whitegrid")
		plt.xlabel("Data sample")
		plt.ylabel("Accuracy")
		plt.title('Accuracy at each data sample')
		#plt.axis([0,350,0,1.0])
		plt.plot(list_iter,acc_list, '-bD', markevery=self.drift)
		#plt.scatter(self.drift,list_iter)
		plt.draw()
		plt.show()
			
	def ozabagging_detection(self):
		#CL=HoeffdingTree()
		CL=NaiveBayes()
		correct=0
		incorrect=0
		#CL=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30)
		clf = OzaBaggingADWINClassifier(base_estimator=CL,n_estimators=10)
		clf.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		for i in range(self.iteration):
			#print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=clf.predict(sample[0])
			#clf = clf.partial_fit(sample[0], sample[1], classes=[0,1])
			#print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
		accuracy=(correct/(correct+incorrect));
		print("Accuracy:",accuracy)
		
	def knnAdwin_detection(self):
		#CL=HoeffdingTree()
		#CL=NaiveBayes()
		correct=0
		incorrect=0
		#CL=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30)
		clf = KNNADWINClassifier(n_neighbors=8, leaf_size=40, max_window_size=1000)
		clf.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		for i in range(self.iteration):
			#print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=clf.predict(sample[0])
			#clf = clf.partial_fit(sample[0], sample[1], classes=[0,1])
			#print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
		accuracy=(correct/(correct+incorrect));
		print("Accuracy:",accuracy)
	def AWE_prediction(self):
		correct=0
		incorrect=0
		#CL=HoeffdingTree()
		CL=NaiveBayes()
		#CL=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30)
		clf = AccuracyWeightedEnsembleClassifier(n_estimators=10, n_kept_estimators=30, base_estimator=CL)
		clf.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		for i in range(self.iteration):
			#print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=clf.predict(sample[0])
			#clf = clf.partial_fit(sample[0], sample[1], classes=[0,1])
			#print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
		accuracy=(correct/(correct+incorrect));
		print("Accuracy:",accuracy)
	def AEE_prediction(self):
		correct=0
		incorrect=0
		#CL=HoeffdingTree()
		#CL=NaiveBayes()
		CL=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30)
		clf = AdditiveExpertEnsembleClassifier(n_estimators=5, base_estimator=CL, beta=0.8, gamma=0.1, pruning='weakest')
		clf.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		for i in range(self.iteration):
			#print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=clf.predict(sample[0])
			#clf = clf.partial_fit(sample[0], sample[1], classes=[0,1])
			#print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
		accuracy=(correct/(correct+incorrect));
		print("Accuracy:",accuracy)
		
	def SRP_prediction(self):
		correct=0
		incorrect=0
		#CL=HoeffdingTree()
		#CL=NaiveBayes()
		CL=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30)
		
		clf = StreamingRandomPatchesClassifier(base_estimator=CL,random_state=1,n_estimators=10)
		clf.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		for i in range(self.iteration):
			#print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=clf.predict(sample[0])
			#clf = clf.partial_fit(sample[0], sample[1], classes=[0,1])
			#print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
		accuracy=(correct/(correct+incorrect));
		print("Accuracy:",accuracy)
		
	def ARF_prediction(self):
		correct=0
		incorrect=0
		#CL=HoeffdingTree()
		#CL=NaiveBayes()
		#CL=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30)
		
		clf = AdaptiveRandomForestClassifier()
		clf.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		for i in range(self.iteration):
			#print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=clf.predict(sample[0])
			#clf = clf.partial_fit(sample[0], sample[1], classes=[0,1])
			#print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
		accuracy=(correct/(correct+incorrect));
		print("Accuracy:",accuracy)
		
	def OSB_prediction(self):
		correct=0
		incorrect=0
		#CL=HoeffdingTree()
		#CL=NaiveBayes()
		#CL=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30)
		clf=OnlineAdaC2Classifier(base_estimator=KNNADWINClassifier(leaf_size=30, max_window_size=1000, metric='euclidean', n_neighbors=5), n_estimators=10, cost_positive=1, cost_negative=0.1, drift_detection=True, random_state=None)
		#clf=OnlineSMOTEBaggingClassifier(base_estimator=CL, n_estimators=10, sampling_rate=1, drift_detection=True, random_state=None)
		clf.fit(self.X_review_dtm1.toarray(),self.review_sentiment_part1,classes=[0,1], sample_weight=None)
		for i in range(self.iteration):
			#print("iteration no: ",i)
			sample=self.ds.next_sample()
			self.y[i]=sample[1]
			self.y_cap[i]=clf.predict(sample[0])
			#clf = clf.partial_fit(sample[0], sample[1], classes=[0,1])
			#print("y_cap=",self.y_cap[i],'y=',self.y[i])
			if(self.y[i]==self.y_cap[i]):
				pred=0
				correct=correct+1
			else:
				pred=1
				incorrect=incorrect+1
		accuracy=(correct/(correct+incorrect));
		print("Accuracy:",accuracy)
		
		
	def create_stream_agarwal(self):
		self.ds=AGRAWALGenerator(classification_function=0, random_state=None, balance_classes=False, perturbation=0.0)
		
		

if __name__=='__main__':
	f = open("Exp_SINGLE_CL_HDDMW_IMDB_KNN.out", 'w')
	sys.stdout = f
	start_time=time.process_time()
	print('start_time: ',start_time)
	#print('ensemble of nb, ddm, mixed')
	st=Potential_drift_detection(4)
	#st.create_stream()
	#st.create_stream_agarwal()
	#st.create_stream_amazon()
	st.create_stream_imdb()
	#st.create_stream_yelp()
	#st.create_stream_mixed()
	#st.ozabagging_detection()
	#st.knnAdwin_detection()
	#st.AWE_prediction()
	#st.AEE_prediction()
	#st.SRP_prediction()
	#st.ARF_prediction()
	#st.OSB_prediction()
	#st.DDM_detection()
	#st.EDDM_detection()
	#st.page_hinkley_detection()
	#st.KSWIN_detection()
	st.HDDM_W_SINGLE_CLASSIFIER_detection()
	#st.four_linear_rates_calculation()
	#st.print_prediction()
	#st.create_stream_amazon()
	end_time=time.process_time()
	execution_time=end_time-start_time
	print("Total Execution time: ",execution_time)
	


	#print(st.ds.n_features)
