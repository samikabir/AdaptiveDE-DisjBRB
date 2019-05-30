#PM_H = 500.4
#PM_M = 35.5
#PM_L = 0.0
 
AQI_H = 500.0
AQI_M = 101.0  
AQI_L = 0.0   

numberOfAntAttributes = 2
#relativeWeight = 1.0   
  
cbd_0 = 1.0
cbd_1 = 0.0  
cbd_2 = 0.0 
cbd_3 = 0.0 
cbd_4 = 1.0
cbd_5 = 0.0  
cbd_6 = 0.0
cbd_7 = 0.0
cbd_8 = 1.0

aqi1 = 1.0
aqi2 = 1.0  
aqi3 = 1.0
aqi4 = 1.0 
aqi5 = 1.0     

def ruleBase(s,c,x):     
    global consequentBeliefDegree
    #global relativeWeight1
    #global relativeWeight2
    temp_consequentBeliefDegree = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]]  
       
    de0 = x[0]/(x[0] + x[1] + x[2])
    de1 = x[1]/(x[0] + x[1] + x[2])
    de2 = x[2]/(x[0] + x[1] + x[2])
    de3 = x[3]/(x[3] + x[4] + x[5]) 
    de4 = x[4]/(x[3] + x[4] + x[5])
    de5 = x[5]/(x[3] + x[4] + x[5])
    de6 = x[6]/(x[6] + x[7] + x[8])
    de7 = x[7]/(x[6] + x[7] + x[8])
    de8 = x[8]/(x[6] + x[7] + x[8])  
        
    consequentBeliefDegree = [de0, de1, de2, de3, de4, de5, de6, de7, de8] 
    attrw1 = x[9]  
    attrw2 = x[10]
    irulewt1 = x[11] 
    irulewt2 = x[12]  
    irulewt3 = x[13]          
    print("Inside ruleBase(x) relativeWeight1 ",attrw1,"relativeWeight2 ",attrw2) 
    #consequentBeliefDegree = [cbd_0, cbd_1, cbd_2, cbd_3, cbd_4, cbd_5, cbd_6, cbd_7, cbd_8]
    for u in range(9):   
        print(consequentBeliefDegree[u])   
    #transformInput1(384.5891688061617)  
    PMH = 0 + (x[14] * 499.4)     #500.4     
    PMM = 0 + (x[15] * 499.4)     #35.5
    PML = 0 + (x[16] * 499.4)     #0    
    transformInput1(s,PMH,PMM,PML)   
    transformInput2(c)    
    calculateMatchingDegreeBrbCnn(attrw1,attrw2, irulewt1, irulewt2, irulewt3)    
    showActivationWeight()
    updateBeliefDegree() 
    result = aggregateER_BrbCnn()
    return result   
  
def deRuleBase(s,c,best):
    global consequentBeliefDegree
    temp_deConsequentBeliefDegree = [best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8]]
    for u in range(9):   
        print("DE Trained Original Belief Degree ",temp_deConsequentBeliefDegree[u])    
     
    de0 = best[0]/(best[0] + best[1] + best[2]) 
    de1 = best[1]/(best[0] + best[1] + best[2]) 
    de2 = best[2]/(best[0] + best[1] + best[2]) 
    de3 = best[3]/(best[3] + best[4] + best[5]) 
    de4 = best[4]/(best[3] + best[4] + best[5])  
    de5 = best[5]/(best[3] + best[4] + best[5]) 
    de6 = best[6]/(best[6] + best[7] + best[8]) 
    de7 = best[7]/(best[6] + best[7] + best[8])
    de8 = best[8]/(best[6] + best[7] + best[8])    
        
    consequentBeliefDegree = [de0, de1, de2, de3, de4, de5, de6, de7, de8] 
    #consequentBeliefDegree = [cbd_0, cbd_1, cbd_2, cbd_3, cbd_4, cbd_5, cbd_6, cbd_7, cbd_8]   
    for u in range(9):    
        print("DE Trained Normalized Belief Degree ",consequentBeliefDegree[u])   
    #transformInput1(384.5891688061617)     
    print("Inside deRuleBase(best) relativeWeight1 ",best[9],"relativeWeight2 ",best[10])  
     
    H_PM = 0 + (best[14] * 499.4)     #500.4     
    M_PM = 0 + (best[15] * 499.4)     #35.5
    L_PM = 0 + (best[16] * 499.4)     #0    
    transformInput1(s,H_PM,M_PM,L_PM)      
    transformInput2(c)    
    calculateMatchingDegreeBrbCnn(best[9], best[10], best[11], best[12], best[13]) 
    showActivationWeight()  
    updateBeliefDegree()       
    deResult = aggregateER_BrbCnn()   
    fo = open("de_best.txt", "w") 
     
    for w_cbd in range(9):
        s_cbd = str(consequentBeliefDegree[w_cbd])   
        fo.write(s_cbd)  
        fo.write("\n") 
  
    for w_cmd in range(9, 14):    
        s_cmd = str(best[w_cmd])     
        fo.write(s_cmd)    
        fo.write("\n")  
   
    s_aauvH = str(H_PM)       
    fo.write(s_aauvH)     
    fo.write("\n")  
    
    s_aauvM = str(M_PM)     
    fo.write(s_aauvM)   
    fo.write("\n") 
    
    s_aauvL = str(L_PM)     
    fo.write(s_aauvL)   
    fo.write("\n")  
 
def transformInput1(i,j,k,l):    
    global H1 
    global M1 
    global L1 
            
    PM_H = j
    PM_M = k 
    PM_L = l
       
    print("Inside transformInput1() Input is ",i,"PM_H ", PM_H,"PM_M ",PM_M," PM_L ",PM_L)
      
    if (i >= PM_H): 
        H1 = 1 
        M1 = 0
        L1 = 0

    elif (i == PM_M):
        H1 = 0 
        M1 = 1
        L1 = 0
 
    elif (i <= PM_L):
        H1 = 0
        M1 = 0
        L1 = 1
       
    elif (i <= PM_H) and (i >= PM_M):
        M1 = (PM_H-i)/(PM_H-PM_M)
        H1 = 1 - M1
        L1 = 0.0 

    elif (i <= PM_M) and (i >= PM_L):
        L1 = (PM_M-i)/(PM_M-PM_L)
        M1 = 1 - L1  
        H1 = 0.0
    print("Inside transformInput1(), H1", H1, "M1 ",M1,"L1 ", L1)

def transformInput2(i):
    global H2   
    global M2 
    global L2
            
    PM_H = 500.4
    PM_M = 35.5
    PM_L = 0.0
       
    print("Inside transformInput2() Input is ",i) 
       
    if (i >= PM_H): 
        H2 = 1 
        M2 = 0 
        L2 = 0

    elif (i == PM_M):
        H2 = 0 
        M2 = 1 
        L2 = 0
  
    elif (i <= PM_L):
        H2 = 0
        M2 = 0
        L2 = 1
        
    elif (i <= PM_H) and (i >= PM_M):
        M2 = (PM_H-i)/(PM_H-PM_M)
        H2 = 1 - M2
        L2 = 0.0 

    elif (i <= PM_M) and (i >= PM_L):
        L2 = (PM_M-i)/(PM_M-PM_L)
        M2 = 1 - L2  
        H2 = 0.0   
    
    print("Inside transformInput2(), H2", H2, "M2 ",M2,"L2 ", L2) 
    
def takeInput():
    global a1     
    temp_a1 = input("Insert value for PM2.5 (between 0 and 500.4 µg/m3): ")
    a1 = float(temp_a1)   
    #transformInput1(a1)  

def calculateMatchingDegreeBrbCnn(aw1,aw2,irw1,irw2,irw3):
    antattrw1 = aw1 
    antattrw2 = aw2
    global initialRuleWeight      
    initialRuleWeight = [irw1, irw2, irw3]     
    increment = 0     
    global matchingDegree 
    matchingDegree = [1.51, 1.51, 1.51]
     
    global trainedMatchingDegree
    trainedMatchingDegree = [1.51, 1.51, 1.51]  
  
    ti1 = [H1, M1, L1]  
    #print("ti1[0] is ")         
    #print(ti1[0])  
    #ti2 = array.array('f', [normalized_cnn_severe_degree, normalized_cnn_mild_degree, normalized_cnn_nominal_degree])
    ti2 = [H2, M2, L2] 
     
    
    for c in range(3):          
        #print(ti1[c]) 
        print("Inside calculateMatchingDegreeBrbCnn() initialRuleWeight[increment] is ",initialRuleWeight[increment])
        matchingDegree[increment] = initialRuleWeight[increment] * (ti1[c] ** antattrw1) * (ti2[c] ** antattrw2)     
        trainedMatchingDegree[increment] = ((ti1[c] ** antattrw1) + (ti2[c] ** antattrw2))
        increment +=1      
    print("Inside calculateMatchingDegreeBrbCnn() relativeWeight1 ",antattrw1,"relativeWeight2 ",antattrw2)   
    #print("Inside calculateMatchingDegreeBrbCnn() best9 relativeWeight1 ",best[9]," best10 relativeWeight2 ",best[10])      
            
def showMatchingDegree():      
    track = 1  
    for counter in range(9):
        track+=1     
  
def showActivationWeight():   
    trace = 1           
    totalWeight = 0 
    totalActivationWeight = 0   
    global activationWeight 
    activationWeight = [1.51, 1.41, 1.45]       
    temp_activationWeight = [1.57, 1.81, 1.92]    
    for x in range(3):    
        totalWeight += matchingDegree[x]           
     
    for counter in range(3):
        print("Inside showActivationWeight() initialRuleWeight[counter] is ",initialRuleWeight[counter])             
        inter = initialRuleWeight[counter] * trainedMatchingDegree[counter]          
        temp_activationWeight[counter] = inter/totalWeight   
            
    for naw in range(3):
        totalActivationWeight += temp_activationWeight[naw]        
       
    for fin in range(3):
        activationWeight[fin] = temp_activationWeight[fin]/totalActivationWeight   
        
def takeCnnOutput():
    global normalized_cnn_severe_degree 
    global normalized_cnn_mild_degree 
    global normalized_cnn_nominal_degree
    
    parser = 0
    #f = open("cnn_prediction.txt", "r") #cnn output
    f = open("cnn_prediction1.txt", "r") #severe 408       
    #f = open("cnn_prediction2.txt", "r") #nominal 36
    #f = open("cnn_prediction3.txt", "r") #mild 117
    if f.mode == 'r':
        #print("reading cnn_prediction.txt file \n") 
        f1 = f.readlines()
         
        for line in f1:  
            if parser == 0: 
                cnn_mild = line
            elif parser == 1:
                cnn_nominal = line
            else: 
                cnn_severe = line
                
            parser +=1    
        
        f.close()    
    else:
        print("Unable to open the file.");
            
    a = float(cnn_mild)
    b = float(cnn_nominal) 
    c = float(cnn_severe)     
    
    mild_degree = a/100    
    nominal_degree = b/100 
    severe_degree = c/100
    
    sum_degree = severe_degree + mild_degree + nominal_degree
  
    normalized_cnn_severe_degree = severe_degree/sum_degree
    normalized_cnn_mild_degree = mild_degree/sum_degree      
    normalized_cnn_nominal_degree = nominal_degree/sum_degree       
    
    if ((normalized_cnn_severe_degree > normalized_cnn_mild_degree) and (normalized_cnn_severe_degree > normalized_cnn_nominal_degree)):
        cnn_pm25 = (150.5 + 349.9*normalized_cnn_severe_degree) + ((150.4*normalized_cnn_mild_degree)/2)
        print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")  

    elif ((normalized_cnn_nominal_degree > normalized_cnn_mild_degree) and (normalized_cnn_nominal_degree > normalized_cnn_severe_degree)):       
        cnn_pm25 = (35.4*(1 - normalized_cnn_nominal_degree)) + ((150.4*normalized_cnn_mild_degree)/2)            
        print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")   

    elif ((normalized_cnn_mild_degree > normalized_cnn_severe_degree) and (normalized_cnn_mild_degree > normalized_cnn_nominal_degree)):    
        if normalized_cnn_severe_degree > normalized_cnn_nominal_degree: 
            cnn_pm25 = (35.5 + 114.9*normalized_cnn_mild_degree) + ((500.4*normalized_cnn_severe_degree)/2)
            print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")  
            
        elif (normalized_cnn_nominal_degree > normalized_cnn_severe_degree): 
            cnn_pm25 = (35.5 + 114.9*normalized_cnn_mild_degree) + ((35.4*normalized_cnn_nominal_degree)/2)     
            print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")
      
 
def updateBeliefDegree():
    update = 0
    sumAntAttr1 = 1
    sumAntAttr2 = 1  
    
    if (H1 + M1 + L1) < 1:
        sumAntAttr1 = H1 + M1 + L1
        update = 1 
      
    if (H2 + M2 + L2) < 1:
        sumAntAttr2 = H2 + M2 + L2
        update = 1 
     
    if update == 1:
        beliefDegreeChangeLevel = (sumAntAttr1 + sumAntAttr2)/numberOfAntAttributes 

        for go in range(9):
            consequentBeliefDegree[go] = beliefDegreeChangeLevel * consequentBeliefDegree[go]
    else: 
        print ("No upgradation of belief degree required.") 
  
def aggregateER_BrbCnn():   
    parse = 0
    move1 = 0 
    move2 = 1  
    move3 = 2 
    action1 = 0
    action2 = 1
    action3 = 2 
    
    global ruleWiseBeliefDegreeSum 
    ruleWiseBeliefDegreeSum = [1.51, 1.51, 1.51]
    
    part11 = 1.51
    part12 = 1.51
    part13 = 1.51
    
    part1 = 1.0
    part2 = 1.0
    value = 1.0
    meu = 1.0
    
    numeratorH1 = 1.0
    numeratorH2 = 1.0
    numeratorH = 1.0
    denominatorH1 = 1.0
    denominatorH = 1.0
    
    numeratorM1 = 1.0  
    numeratorM = 1.0
    
    numeratorL1 = 1.0
    numeratorL = 1.0
     
    utilityScoreH = 1.0
    utilityScoreM = 0.5
    utilityScoreL = 0.0
    crispValue = 1.0
    degreeOfIncompleteness = 1.0
    utilityMax = 1.0 
    utilityMin = 1.0
    utilityAvg = 1.0
    
    global aqi
    
    for s in range(9): 
        print("Inside aggregateER)BrbCNN() consequentBeliefDegree: ",consequentBeliefDegree[s])
     
    for t in range(3): 
        parse = t * 3   
        ruleWiseBeliefDegreeSum[t] = consequentBeliefDegree[parse] + consequentBeliefDegree[parse+1] + consequentBeliefDegree[parse+2]
 
    for rule in range(3):  
        part11 *= (activationWeight[rule] * consequentBeliefDegree[move1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))         
        move1 += 3 
  
    for rule in range(3):
        part12 *= (activationWeight[rule] * consequentBeliefDegree[move2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move2 += 3 
 
    for rule in range(3):
        part13 *= (activationWeight[rule] * consequentBeliefDegree[move3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move3 += 3

    part1 = (part11 + part12 + part13)
    
    for rule in range(3):
        part2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule])) 
    
    value = part1 - part2 
    
    meu = 1/value 
 
    for rule in range(3):
        numeratorH1 *= (activationWeight[rule] * consequentBeliefDegree[action1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action1 += 3

    for rule in range(3):
        numeratorH2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))              
      
    numeratorH = meu * (numeratorH1 - numeratorH2) 
    
    for rule in range(3):  
        denominatorH1 *= (1 - activationWeight[rule])        
 
    denominatorH = 1 - (meu * denominatorH1)
    
    aggregatedBeliefDegreeH = (numeratorH/denominatorH)
    
    for rule in range(3):
        numeratorM1 *= (activationWeight[rule] * consequentBeliefDegree[action2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action2 += 3 

    numeratorM = meu * (numeratorM1 - numeratorH2) 
    aggregatedBeliefDegreeM = (numeratorM/denominatorH)  
    
    for rule in range(3):
        numeratorL1 *= (activationWeight[rule] * consequentBeliefDegree[action3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action3 += 3
     
    numeratorL = meu * (numeratorL1 - numeratorH2)
    aggregatedBeliefDegreeL = (numeratorL/denominatorH) 
    
    if (aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL) == 1:
        crispValue = (aggregatedBeliefDegreeH * utilityScoreH) + (aggregatedBeliefDegreeM * utilityScoreM) + (aggregatedBeliefDegreeL * utilityScoreL)
        brbH = aggregatedBeliefDegreeH
        brbM = aggregatedBeliefDegreeM
        brbL = aggregatedBeliefDegreeL       
        
        print ("\n BRB-CNN integrated Belief Degree for Hazardous AQI: ",aggregatedBeliefDegreeH,"\n")
        print ("\n BRB-CNN integrated Belief Degree for Unhealthy AQI: ",aggregatedBeliefDegreeM,"\n")
        print ("\n BRB-CNN integrated Belief Degree for Good AQI: ",aggregatedBeliefDegreeL,"\n")
        #cout << "brbH: " << brbH << " brbM: " << brbM << " brbL: " << brbL <<endl;    
 
    else:         
        degreeOfIncompleteness = 1 - (aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)
        
        utilityMax = ((aggregatedBeliefDegreeH + degreeOfIncompleteness) * utilityScoreH + (aggregatedBeliefDegreeM*utilityScoreM) + (aggregatedBeliefDegreeL*utilityScoreL))
        
        utilityMin = (aggregatedBeliefDegreeH*utilityScoreH) + (aggregatedBeliefDegreeM*utilityScoreM) + (aggregatedBeliefDegreeL + degreeOfIncompleteness) * utilityScoreL
        
        utilityAvg = (utilityMax + utilityMin)/2  
         
        print ("BRB-CNN integrated Belief Degrees considering degree of Incompleteness: ")  
        
        finalAggregatedBeliefDegreeH = aggregatedBeliefDegreeH/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)  
         
        finalAggregatedBeliefDegreeM = aggregatedBeliefDegreeM/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)
        
        finalAggregatedBeliefDegreeL = aggregatedBeliefDegreeL/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL) 
          
        brbH = finalAggregatedBeliefDegreeH
        brbM = finalAggregatedBeliefDegreeM 
        brbL = finalAggregatedBeliefDegreeL       
            
        if (finalAggregatedBeliefDegreeH > finalAggregatedBeliefDegreeM) and (finalAggregatedBeliefDegreeH > finalAggregatedBeliefDegreeL):
            aqi = (201 + 299*finalAggregatedBeliefDegreeH) + ((200*finalAggregatedBeliefDegreeM)/2)
            print ("AQI predicted by BRB-CNN:",aqi)    
            
        elif (finalAggregatedBeliefDegreeL > finalAggregatedBeliefDegreeM) and (finalAggregatedBeliefDegreeL > finalAggregatedBeliefDegreeH): 
            aqi = (100*(1 - finalAggregatedBeliefDegreeL)) + ((200*finalAggregatedBeliefDegreeM)/2) 
            print ("AQI predicted by BRB-CNN:",aqi)
  
        elif (finalAggregatedBeliefDegreeM > finalAggregatedBeliefDegreeH) and (finalAggregatedBeliefDegreeM > finalAggregatedBeliefDegreeL):
            if finalAggregatedBeliefDegreeH > finalAggregatedBeliefDegreeL:
                aqi = (101 + 99*finalAggregatedBeliefDegreeM) + ((500*finalAggregatedBeliefDegreeH)/2)
                print ("AQI predicted by BRB-CNN: ",aqi)
      
            elif (finalAggregatedBeliefDegreeL > finalAggregatedBeliefDegreeH):   
                aqi = (101 + 99*finalAggregatedBeliefDegreeM) + ((100*finalAggregatedBeliefDegreeL)/2)
                print ("AQI predicted by BRB-CNN:",aqi)  
          
        print("aqi ",aqi)     
                
        if aqi >= 301: 
            aqi6 = (aqi- 301)/199.0  
 
        elif (aqi >= 201)and (aqi <= 300.9999999999): 
            aqi6 = (aqi- 201)/99.0     

        elif (aqi >= 151)and (aqi <= 200.9999999999):
            aqi6 = (aqi- 151)/49.0 

        elif((aqi >= 101)and (aqi <= 150.9999999999)): 
            aqi6 = (aqi- 101)/49.0   

        elif((aqi >= 51)and (aqi <= 100.9999999999)): 
            aqi6 = (aqi- 51)/49.0    
 
        elif(aqi <= 50.9999999999):    
            aqi6 = (aqi/49.0)    
             
        print("aqi6 ",aqi6) 
        print ("BRB-CNN integrated Belief Degree for Hazardous AQI:",finalAggregatedBeliefDegreeH*aqi6)   
        print ("BRB-CNN integrated Belief Degree for Very Unhealthy AQI:",finalAggregatedBeliefDegreeH*(1-aqi6)) 
        print ("BRB-CNN integrated Belief Degree for Unhealthy AQI: ",finalAggregatedBeliefDegreeM*aqi6)
        print ("BRB-CNN integrated Belief Degree for Unhealthy (Sensitive Groups) AQI:",finalAggregatedBeliefDegreeM*(1-aqi6)) 
        print ("BRB-CNN integrated Belief Degree for Moderate AQI:",finalAggregatedBeliefDegreeL*aqi6) 
        print ("BRB-CNN integrated Belief Degree for Good AQI:",finalAggregatedBeliefDegreeL*(1-aqi6))
        
        return aqi

#def getAQI(x):     
#    cbd_de0 = x[0]
#    cbd_de1 = x[1] 
#    cbd_de2 = x[2]
#    ruleBase()
     
    #aqi = x[0] + x[1] + x[2] + x[3]   
    #print("Diff Evo BRB/CNN AQI is ",aqi) 
#   return cbd_de0 + cbd_de1 + cbd_de2   
   
#def main():
#    ruleBase()       
#    takeInput()  
    #showTransformedInput() unnecessary      
#    takeCnnOutput() 
#    calculateMatchingDegreeBrbCnn() 
    #showMatchingDegree() unnecessary
#    showActivationWeight()   
#    updateBeliefDegree()    
#    aggregateER_BrbCnn()
    #getAQI(x) unnecessary

#main()     