import numpy as np 
import array
from random import *
import dbrb2 
import dbrb3   
import dbrb4  
import dbrb5
import dbrb6
import bestBRB2 
import bestBRB3
import bestBRB4
import bestBRB5
import bestBRB6 
import math 
import Conj_a_BRB_CR as ConjBRB_CR  
import Conj_a_BRB_F as ConjBRB_F
import Disj_a_BRB_CR as DisjBRB_CR
import Disj_a_BRB_F as DisjBRB_F   
#len(bounds)      

f1 = open("sensor.txt", "r")
f2 = open("cnn.txt", "r")
f3 = open("aqi.txt", "r")
  
sensor = [10.35] * 150 
cnn = [10.35] * 150
aqi = [10.35] * 150 
refvl = [3] * 5
refValList = [3] * 5
msel = [3.23] * 5
setList = [0] * 10
setList[0] = 3
#cnn = array.array('f', [10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, #10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32])

#aqi = array.array('f', [10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32])
    #f = open("cnn_prediction2.txt", "r") #nominal 36 
    #f = open("cnn_prediction3.txt", "r") #mild 117
tk1 = 0
tk2 = 0 
tk3 = 0   
  
if f1.mode == 'r':  
        #print("reading cnn_prediction.txt file \n")  
    f11 = f1.readlines()
    
    for line in f11:   
        sensor[tk1] = float(line)
        tk1 += 1 
    #print(sensor[12])

else:
    print("Unable to open the file sensor.txt")
    
    
if f2.mode == 'r':
        #print("reading cnn_prediction.txt file \n")  
    f21 = f2.readlines() 
     
    for line in f21:   
        cnn[tk2] = float(line)
        tk2 += 1
    #print(cnn[11])     

else:
    print("Unable to open the file cnn.txt") 
            

if f3.mode == 'r':
        #print("reading cnn_prediction.txt file \n")  
    f31 = f3.readlines()
    
    for line in f31:   
        aqi[tk3] = float(line)
        tk3 += 1
            
    #print(aqi[13])           

else:
    print("Unable to open the file aqi.txt")     
fo = open("de_mse.txt", "w") 
rec_test_aqi = open("predicted_aqi_mse.txt", "w")
cross_factor = open("crossover_factor.txt", "w")
mut_factor = open("mutation_factor.txt", "w") 
   
def de(fobj, dimensions, bounds, popsize=40, its=1000): #popsize = 170 
    mut=0.8 #F
    crossp=0.7 #CR       
    #dimensions = 32 #original 17 against H, M, L       
    global best
    cross_factor.write("de starts...crossover factor is ")
    cross_factor.write("\n")
    mut_factor.write("de starts...mutation factor is ") 
    mut_factor.write("\n")   
    pop = np.random.rand(popsize, dimensions) 
    min_b, max_b = np.asarray(bounds).T  
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff  
    #fitness = (70-np.asarray([fobj(ind) for ind in pop_denorm]))**2 #BRB_DE
    fitness = (np.asarray([fobj(ind) for ind in pop_denorm])) #BRB_DE  
    best_idx = np.argmin(fitness)  
    best = pop_denorm[best_idx] 
    for i in range(its):  
        subr = 0.00
        pc_t = 0.00
        pc = 0.00
        fc = 0.00  
        diffres = 0.00  
        end = 1  
        for j in range(popsize):    
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1) 
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff   
            #f = (70-fobj(trial_denorm))**2 #BRB_DE  
            f = (fobj(trial_denorm)) #BRB_DE 
            if f < fitness[j]:  
                end = 0 
                subr += (trial_denorm - pop_denorm[j])**2                 
                diffres += ((f - fitness[j])/1000)**2   
                print("diffress = ",diffres)
                fitness[j] = f   
                pop[j] = trial
                pop_denorm[j] = trial_denorm   
                if f < fitness[best_idx]:  
                    best_idx = j 
                    best = trial_denorm
        if i == 99: 
            fo.write("After 100 its, MSE is ")
            fo.write(str(fitness[best_idx])) 
            fo.write("\n")
        elif i == 199:
            fo.write("After 200 its, MSE is ")
            fo.write(str(fitness[best_idx]))  
            fo.write("\n")
        elif i == 299:
            fo.write("After 300 its, MSE is ")
            fo.write(str(fitness[best_idx])) 
            fo.write("\n")
        elif i == 399: 
            fo.write("After 400 its, MSE is ") 
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 499:
            fo.write("After 500 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 599:
            fo.write("After 600 its, MSE is ")
            fo.write(str(fitness[best_idx])) 
            fo.write("\n")
        elif i == 699:
            fo.write("After 700 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 799:
            fo.write("After 800 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 899: 
            fo.write("After 900 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 999: 
            fo.write("After 1000 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        
        if (end == 0):    
            for r in range(dimensions):        
                pc_t += subr[r]
                print("pc_t+=", pc_t)
        #yield best
        print("summation of subr, pc_t =", pc_t) 
        pc = math.sqrt(pc_t/popsize)   
        print("summation of subr, pc =", pc) 
        print("summation of diffres =", diffres)
        fc = math.sqrt(diffres/popsize) 
        print("summation of diffres, fc =", fc) 
        d11_t = (1 + pc) * (math.exp(-pc))   
        d11 = 1 - d11_t 
        print("value of d11 =", d11) 
        d12_t = (1 + fc) * (math.exp(-fc))     
        d12 = 1 - d12_t    
        print("value of d12 =", d12)    
        d21 = 2 * d11
        d22 = 2 * d12   
        print("value of d21 = ",d21) 
        print("value of d21 = ",d22) 
        crossp = ConjBRB_CR.ruleBase(d11, d12)
        print("After ConjBRB_CR BRBaDE, crossp value is", crossp)
        cross_factor.write(str(crossp))     
        cross_factor.write("\n")   
        #crossp = DisjBRB_CR.ruleBase(d11, d12) 
        #print("After DisjBRB_CR BRBaDE, crossp value is", crossp)     
        mut = ConjBRB_F.ruleBase(d21, d22)
        print("After ConjBRB_F BRBaDE, mut value is", mut) 
        mut_factor.write(str(mut))    
        mut_factor.write("\n")            
        #mut = DisjBRB_F.ruleBase(d21, d22)
        #print("After DisjBRB_F BRBaDE, mut value is", mut)     
        yield best, fitness[best_idx]     
         #yield best, fitness[best_idx] 
        #yield min_b + pop * diff, fitness, best_idx           
#it = list(de(lambda x: x**2, bounds=[(1, 20)]))                                    
def fobj2(x):   
    sum = 0   
    for go in range(120):            
        pred_aqi = dbrb2.ruleBase(sensor[go], cnn[go], x) 
        sum += (aqi[go] - pred_aqi)**2    
    return sum/120            
  
def fobj3(x):
    sum = 0   
    for go in range(120):            
        pred_aqi = dbrb3.ruleBase(sensor[go], cnn[go], x)
        sum += (aqi[go] - pred_aqi)**2    
    return sum/120            
           
def fobj4(x):   
    sum = 0     
    for go in range(120):            
        pred_aqi = dbrb4.ruleBase(sensor[go], cnn[go], x)
        sum += (aqi[go] - pred_aqi)**2      
    return sum/120    

def fobj5(x): 
    sum = 0      
    for go in range(120):            
        pred_aqi = dbrb5.ruleBase(sensor[go], cnn[go], x)
        sum += (aqi[go] - pred_aqi)**2      
    return sum/120   

def fobj6(x): 
    sum = 0      
    for go in range(120):            
        pred_aqi = dbrb6.ruleBase(sensor[go], cnn[go], x)
        sum += (aqi[go] - pred_aqi)**2      
    return sum/120     
 
it = list(de(fobj3, 17, bounds=[(0, 1)]))    
print(it[-1])        
      
for g in range(17):       
    print("best[0-16] ", best[g])                      

test_MSE_3_0 = 0       
for parse in range(120, 150):        
    test_pred_aqi = bestBRB3.deRuleBase(sensor[parse], cnn[parse], best) 
    test_MSE_3_0 += (aqi[parse] - test_pred_aqi)**2       
    rec_test_aqi.write(str(test_pred_aqi))
    rec_test_aqi.write("\n")   
     
rec_test_aqi.write("msel[0], test_MSE_3_0 against 3 Ref Val during JOPS is ")
rec_test_aqi.write(str(test_MSE_3_0))  
rec_test_aqi.write("\n") 

mse_final_3 = (test_MSE_3_0/30)  
rec_test_aqi.write("msel[0], MSE of Test Dataset against 3 Ref Val during JOPS is ")
rec_test_aqi.write(str(mse_final_3))
rec_test_aqi.write(" ")     
msel[0] =  mse_final_3
setList[0] = mse_final_3 
t_refv = refvl[0]       
       
for g in range(refvl[0]):  
    if(random() < 0.25):
        t_refv += 1
refvl[1] = t_refv       
         
if (refvl[1] == 3):
    refValList[1] = 3
    msel[1] =  msel[0]
    setList[1] = setList[0] 
    rec_test_aqi.write("msel[1] unchanged, MSE of Test Dataset against 3 Ref Val during JOPS. ")
      
elif (refvl[1] == 4):      
    it = list(de(fobj4, 22, bounds=[(0, 1)]))   
    print(it[-1])
    
    test_MSE_4_1 = 0  
    
    for parse in range(120, 150):                 
        test_pred_aqi = bestBRB4.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_4_1 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")    
     
    mse_final_4 = test_MSE_4_1/30  
    rec_test_aqi.write("msel[1], MSE of Test Dataset against 4 Ref Val during JOPS is ")  
    rec_test_aqi.write(str(mse_final_4))
    rec_test_aqi.write(" ")
    refValList[1] = 4
    msel[1] =  mse_final_4     
    setList[1] = mse_final_4
    
elif (refvl[1] == 5):     
    it = list(de(fobj5, 27, bounds=[(0, 1)]))   
    print(it[-1])
    
    test_MSE_5_1 = 0 
    
    for parse in range(120, 150):             
        test_pred_aqi = bestBRB5.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_5_1 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_5 = test_MSE_5_1/30   
    rec_test_aqi.write("msel[1], MSE of Test Dataset against 5 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_5))
    rec_test_aqi.write(" ")
    refValList[1] = 5
    msel[1] =  mse_final_5
    setList[1] = mse_final_5
      
elif (refvl[1] == 6):        
    it = list(de(fobj6, 32, bounds=[(0, 1)]))   
    print(it[-1])
    
    test_MSE_6_1 = 0
    
    for parse in range(120, 150):                 
        test_pred_aqi = bestBRB6.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_6_1 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
      
    mse_final_6 = test_MSE_6_1/30  
    rec_test_aqi.write("msel[1], MSE of Test Dataset against 6 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_6))
    rec_test_aqi.write(" ")
    refValList[1] = 6       
    msel[1] =  mse_final_6    
    setList[1] = mse_final_6  
    
if (msel[0] >= msel[1]):
    if (refValList[1] >= refValList[0]):
        t_ref = refValList[0]
        for r in range(refValList[0]):
            if (random() < 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2): 
            t_ref = 2    
        refValList[2] = t_ref               
    else:   
        t_ref = 0   
        for r in range(refValList[0]):
            if (random() >= 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2):
            t_ref = 2
        refValList[2] = t_ref              
        
elif (msel[1] > msel[0]):
    if (refValList[0] >= refValList[1]):
        t_ref = refValList[1]
        for r in range(refValList[1]):
            if (random() < 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2):
            t_ref = 2
        refValList[2] = t_ref
    else:
        t_ref = 0    
        for r in range(refValList[1]):
            if (random() >= 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2):
            t_ref = 2
        refValList[2] = t_ref    
           
if (refValList[2] == 2):  
    it = list(de(fobj2, 12, bounds=[(0, 1)]))    
    print(it[-1])
    
    test_MSE_2_2 = 0
    
    for parse in range(120, 150):               
        test_pred_aqi = bestBRB2.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_2_2 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
     
    mse_final_2 = test_MSE_2_2/30    
    rec_test_aqi.write("msel[2], MSE of Test Dataset against 2 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_2)) 
    rec_test_aqi.write(" ")
    msel[2] =  mse_final_2
                   
elif (refValList[2] == 3): 
    it = list(de(fobj3, 17, bounds=[(0, 1)]))   
    print(it[-1]) 
    
    test_MSE_3_2 = 0
    
    for parse in range(120, 150):               
        test_pred_aqi = bestBRB3.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_3_2 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_3 = test_MSE_3_2/30 
    rec_test_aqi.write("msel[2], MSE of Test Dataset against 3 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_3))
    rec_test_aqi.write(" ")
    msel[2] =  mse_final_3       
           
elif (refValList[2] == 4):        
    it = list(de(fobj4, 22, bounds=[(0, 1)]))   
    print(it[-1])
    
    test_MSE_4_2 = 0
    
    for parse in range(120, 150):              
        test_pred_aqi = bestBRB4.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_4_2 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_4 = test_MSE_4_2/30 
    rec_test_aqi.write("msel[2], MSE of Test Dataset against 4 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_4))
    rec_test_aqi.write(" ")
    msel[2] =  mse_final_4       
    
elif (refValList[2] == 5):     
    it = list(de(fobj5, 27, bounds=[(0, 1)]))   
    print(it[-1])
    
    test_MSE_5_2 = 0  
    
    for parse in range(120, 150):            
        test_pred_aqi = bestBRB5.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_5_2 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_5 = test_MSE_5_2/30 
    rec_test_aqi.write("msel[2], MSE of Test Dataset against 5 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_5)) 
    rec_test_aqi.write(" ")
    msel[2] =  mse_final_5
      
elif (refValList[2] == 6):         
    it = list(de(fobj6, 32, bounds=[(0, 1)]))   
    print(it[-1])  
    
    test_MSE_6_2 = 0  
    
    for parse in range(120, 150):               
        test_pred_aqi = bestBRB6.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_6_2 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
     
    mse_final_6 = test_MSE_6_2/30 
    rec_test_aqi.write("msel[2], MSE of Test Dataset against 6 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_6))  
    rec_test_aqi.write(" ")   
    msel[2] =  mse_final_6          
      
if (msel[1] >= msel[2]):
    if (refValList[2] >= refValList[1]):
        t_ref = refValList[1]
        for r in range(refValList[1]):
            if (random() < 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2):
            t_ref = 2
        refValList[3] = t_ref 
    else:  
        t_ref = 0   
        for r in range(refValList[1]):
            if (random() >= 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6
        elif (t_ref < 2):
            t_ref = 2
        refValList[3] = t_ref  
             
elif (msel[2] > msel[1]): 
    if (refValList[1] >= refValList[2]):
        t_ref = refValList[2]
        for r in range(refValList[2]):
            if (random() < 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6
        elif (t_ref < 2):
            t_ref = 2          
        refValList[3] = t_ref
    else:
        t_ref = 0   
        for r in range(refValList[2]):
            if (random() >= 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6
        elif (t_ref < 2):
            t_ref = 2
        refValList[3] = t_ref                  

if (refValList[3] == 2):  
    it = list(de(fobj2, 12, bounds=[(0, 1)]))   
    print(it[-1])    
    
    test_MSE_2_3 = 0 
    
    for parse in range(120, 150):              
        test_pred_aqi = bestBRB2.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_2_3 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
     
    mse_final_2 = test_MSE_2_3/30    
    rec_test_aqi.write("msel[3], MSE of Test Dataset against 2 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_2)) 
    rec_test_aqi.write(" ")
    msel[3] =  mse_final_2            
        
elif (refValList[3] == 3): 
    it = list(de(fobj3, 17, bounds=[(0, 1)]))   
    print(it[-1])
    
    test_MSE_3_3 = 0
    
    for parse in range(120, 150):                
        test_pred_aqi = bestBRB3.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_3_3 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
     
    mse_final_3 = test_MSE_3_3/30 
    rec_test_aqi.write("msel[3], MSE of Test Dataset against 3 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_3))
    rec_test_aqi.write(" ")
    msel[3] =  mse_final_3       
     
elif (refValList[3] == 4):        
    it = list(de(fobj4, 22, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_4_3 = 0  
    for parse in range(120, 150):             
        test_pred_aqi = bestBRB4.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_4_3 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_4 = test_MSE_4_3/30 
    rec_test_aqi.write("msel[3], MSE of Test Dataset against 4 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_4))
    rec_test_aqi.write(" ")
    msel[3] =  mse_final_4       
    
elif (refValList[3] == 5):     
    it = list(de(fobj5, 27, bounds=[(0, 1)]))   
    print(it[-1])
    test_MSE_5_3 = 0 
    for parse in range(120, 150):              
        test_pred_aqi = bestBRB5.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_5_3 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_5 = test_MSE_5_3/30 
    rec_test_aqi.write("msel[3], MSE of Test Dataset against 5 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_5)) 
    rec_test_aqi.write(" ")
    msel[3] =  mse_final_5
      
elif (refValList[3] == 6):         
    it = list(de(fobj6, 32, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_6_3 = 0     
    for parse in range(120, 150):               
        test_pred_aqi = bestBRB6.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_6_3 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
     
    mse_final_6 = test_MSE_6_3/30 
    rec_test_aqi.write("msel[3], MSE of Test Dataset against 6 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_6))  
    rec_test_aqi.write(" ")
    msel[3] =  mse_final_6       

if (msel[2] >= msel[3]):
    if (refValList[3] >= refValList[2]):
        t_ref = refValList[2]
        for r in range(refValList[2]):
            if (random() < 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2):
            t_ref = 2
        refValList[4] = t_ref 
    else:  
        t_ref = 0   
        for r in range(refValList[2]):
            if (random() >= 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6
        elif (t_ref < 2):
            t_ref = 2
        refValList[4] = t_ref 
             
elif (msel[3] > msel[2]):  
    if (refValList[2] >= refValList[3]):
        t_ref = refValList[3]
        for r in range(refValList[3]):
            if (random() < 0.25):
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2):
            t_ref = 2
        refValList[4] = t_ref 
    else:  
        t_ref = 0    
        for r in range(refValList[3]):
            if (random() >= 0.25): 
                t_ref += 1
        if (t_ref > 6):
            t_ref = 6  
        elif (t_ref < 2):
            t_ref = 2
        refValList[4] = t_ref          
if (refValList[4] == 2):
    it = list(de(fobj2, 12, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_2_4 = 0     
    for parse in range(120, 150):             
        test_pred_aqi = bestBRB2.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_2_4 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_2 = test_MSE_2_4/30 
    rec_test_aqi.write("msel[4], MSE of Test Dataset against 2 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_2))
    rec_test_aqi.write(" ")
    msel[4] =  mse_final_2               
        
elif (refValList[4] == 3):
    it = list(de(fobj3, 17, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_3_4 = 0  
    for parse in range(120, 150):             
        test_pred_aqi = bestBRB3.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_3_4 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_3 = test_MSE_3_4/30 
    rec_test_aqi.write("msel[4], MSE of Test Dataset against 3 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_3))
    rec_test_aqi.write(" ")
    msel[4] =  mse_final_3       
     
elif (refValList[4] == 4):        
    it = list(de(fobj4, 22, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_4_4 = 0    
    for parse in range(120, 150):             
        test_pred_aqi = bestBRB4.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_4_4 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_4 = test_MSE_4_4/30 
    rec_test_aqi.write("msel[4], MSE of Test Dataset against 4 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_4))
    rec_test_aqi.write(" ")
    msel[4] =  mse_final_4       
    
elif (refValList[4] == 5):     
    it = list(de(fobj5, 27, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_5_4 = 0
    for parse in range(120, 150):               
        test_pred_aqi = bestBRB5.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_5_4 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_5 = test_MSE_5_4/30  
    rec_test_aqi.write("msel[4], MSE of Test Dataset against 5 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_5)) 
    rec_test_aqi.write(" ")
    msel[4] =  mse_final_5 
      
elif (refValList[4] == 6):           
    it = list(de(fobj6, 32, bounds=[(0, 1)]))   
    print(it[-1])  
    test_MSE_6_4 = 0     
    for parse in range(120, 150):               
        test_pred_aqi = bestBRB6.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_6_4 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
     
    mse_final_6 = test_MSE_6_4/30 
    rec_test_aqi.write("msel[4], MSE of Test Dataset against 6 Ref Val during JOPS is ")
    rec_test_aqi.write(str(mse_final_6))  
    rec_test_aqi.write(" ")
    msel[4] =  mse_final_6   
min_mse = min(msel) 

for i in range(5):
    if (min_mse == msel[i]):
        index = i 
    
final_rv = refValList[index]           
  
#print("Final Number of referential values after JOPS is ", final_rv) 

if (final_rv == 2):
    it = list(de(fobj2, 12, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_rv2 = 0
    for parse in range(120, 150):                  
        test_pred_aqi = bestBRB2.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_rv2 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_2 = test_MSE_rv2/30 
    rec_test_aqi.write("MSE of Test Dataset after JOPS against 2 Ref Val is ")
    rec_test_aqi.write(str(mse_final_2))
    #msel[4] =  mse_final_2               
        
elif (final_rv == 3):
    it = list(de(fobj3, 17, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_rv3 = 0 
    for parse in range(120, 150):              
        test_pred_aqi = bestBRB3.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_rv3 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")    
    
    mse_final_3 = test_MSE_rv3/30 
    rec_test_aqi.write("MSE of Test Dataset after JOPS against 3 Ref Val is ")
    rec_test_aqi.write(str(mse_final_3))
    #msel[4] =  mse_final_3        
     
elif (final_rv == 4):         
    it = list(de(fobj4, 22, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_rv4 = 0    
    for parse in range(120, 150):             
        test_pred_aqi = bestBRB4.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_rv4 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
    
    mse_final_4 = test_MSE_rv4/30 
    rec_test_aqi.write("MSE of Test Dataset after JOPS against 4 Ref Val is ")
    rec_test_aqi.write(str(mse_final_4))
    #msel[4] =  mse_final_4        
    
elif (final_rv == 5):     
    it = list(de(fobj5, 27, bounds=[(0, 1)]))   
    print(it[-1]) 
    test_MSE_rv5 = 0   
    for parse in range(120, 150):            
        test_pred_aqi = bestBRB5.deRuleBase(sensor[parse], cnn[parse], best)
        test_MSE_rv5 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")    
    
    mse_final_5 = test_MSE_rv5/30 
    rec_test_aqi.write("MSE of Test Dataset after JOPS against 5 Ref Val  is ")
    rec_test_aqi.write(str(mse_final_5)) 
    #msel[4] =  mse_final_5 
      
elif (final_rv == 6):           
    it = list(de(fobj6, 32, bounds=[(0, 1)]))   
    print(it[-1])  
    test_MSE_rv6 = 0     
    for parse in range(120, 150):               
        test_pred_aqi = bestBRB6.deRuleBase(sensor[parse], cnn[parse], best) 
        test_MSE_rv6 += (aqi[parse] - test_pred_aqi)**2      
        rec_test_aqi.write(str(test_pred_aqi))
        rec_test_aqi.write("\n")   
     
    mse_final_6 = test_MSE_rv6/30 
    rec_test_aqi.write("MSE of Test Dataset after JOPS against 6 Ref Val is ")
    rec_test_aqi.write(str(mse_final_6))  
    #msel[4] =  mse_final_6
    
print("Final Number of referential values after JOPS is ", final_rv)      