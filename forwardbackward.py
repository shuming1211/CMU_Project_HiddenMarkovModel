import sys
import csv
import math
import numpy as np

test_input = sys.argv[1]
index_to_word = sys.argv[2]
index_to_tag = sys.argv[3]
hmmprior = sys.argv[4]
hmmemit = sys.argv[5]
hmmtrans = sys.argv[6]
predicted_file = sys.argv[7]
metric_file = sys.argv[8]

#
wordlist = []
with open(index_to_word,'r') as f:
    for row in f:
        row = row.strip('\n')
        row = row.strip('\r')
        wordlist.append(row)
f.close

#
taglist = []
with open(index_to_tag,'r') as f:
    for row in f:
        row = row.strip('\n')
        row = row.strip('\r')
        taglist.append(row)
f.close

#
with open(test_input,'r') as f:
    word_tag = []    
    for row in f:
        temp = []
        row = row.strip('\n')
        row = row.strip('\r')
        raw_data = row.split(" ")
        
        for element in raw_data:
            (word,tag) = element.split("_") 
            temp.append([wordlist.index(word),taglist.index(tag)])
        word_tag.append(temp)

#       
def importMatrices(hmmprior,hmmemit,hmmtrans):
    pi = []
    with open(hmmprior,'r') as f:
        for row in f:
            row = row.strip('\n')
            pi.append(float(row))
    f.close
    pi = np.array(pi)
    
    A = []
    with open(hmmtrans,'r') as f:
        for row in f:
            row = row.strip()
            raw_data = row.split(" ")
            temp = []
            for item in raw_data:
                temp.append(float(item))
            A.append(temp)
    f.close   
    A = np.array(A)
    
    B = []
    with open(hmmemit,'r') as f:
        for row in f:
            row = row.strip()
            raw_data = row.split(" ")
            temp = []
            for item in raw_data:
                temp.append(float(item))
            B.append(temp)
    f.close
    B = np.array(B)
    
    return pi, A, B
    
def forward(sentence, pi, A, B): #sentences
    alpha = []
     
    T = len(sentence)
    for t in range(T):
        if t == 0:
            temp = np.zeros(len(taglist))
            for j in range(len(taglist)):
                x = sentence[t][0]                 
                temp[j] = (float(pi[j]) * float(B[j][x]))
                
        else:
            temp = np.zeros(len(taglist))
            for j in range(len(taglist)):
                x = sentence[t][0] 
                temp[j] = B[j][x] * np.dot(A[:,j], alpha[-1]) 
        
        if t == T-1:
            alpha_lastrow = temp
        alpha.append(temp)
    
    alpha = np.array(alpha)
       
    return alpha, alpha_lastrow

def backward(sentence, pi, A, B):    
    beta = []

    T = len(sentence)   
    for t in range(T-1,-1,-1):
        temp = np.zeros(len(taglist))
        if t == T-1:
            for j in range(len(taglist)):
                temp[j] = 1
        else:
            for j in range(len(taglist)): 
                x = sentence[t+1][0]
                count = beta[-1]*A[j]
                temp[j] = np.dot(B[:,x],count)
        beta.append(temp)
        
    beta = np.array(beta)        
    return beta
    
def calculate(sentence, pi, A, B):  
    predict_tag = []

    alpha,alpha_lastrow = forward(sentence, pi, A, B)
    beta = backward(sentence, pi, A, B)
    sum_alpha_lastrow = sum(alpha_lastrow)
    log_p = math.log(sum_alpha_lastrow)
    
    p = np.multiply(alpha,beta)
    
    for element in p:
        max_element = max(element)
        tag_index = list(element).index(max_element)
        tag = taglist[tag_index]
        predict_tag.append(tag)
   
    return predict_tag,log_p   # return tags for a single sentence 
  
    
def write(word_tag, hmmprior, hmmemit, hmmtrans,predicted_file,metric_file):
     
    wr_predict = open(predicted_file,'w')
    wr_metric = open(metric_file,'w')
    pi, A, B = importMatrices(hmmprior,hmmemit,hmmtrans)
    log = []
    count_correct = 0
    count_total = 0
    
    for sentence in word_tag:
        predict_tags, log_p = calculate(sentence, pi, A, B)
        
        for i in range(len(sentence)):
            
            word = wordlist[sentence[i][0]]
            tag_predict = predict_tags[i]
            count_total += 1
            
            if tag_predict == taglist[sentence[i][1]]:
                count_correct += 1
                
            if i == len(sentence) - 1:
                wr_predict.write(str(word) + '_' + str(tag_predict))
            else:
                wr_predict.write(str(word) + '_' + str(tag_predict) + ' ')
        wr_predict.write('\n')    
        log.append(log_p)
    log_average = sum(log) / len(word_tag) 

    accuracy = float(count_correct) / count_total
    wr_metric.write("Average Log-Likelihood: " + str(log_average) + '\n')
    wr_metric.write("Accuracy: " + str(accuracy))
    
if __name__ == "__main__":    
    write(word_tag, hmmprior, hmmemit, hmmtrans,predicted_file,metric_file)    
    

    
    