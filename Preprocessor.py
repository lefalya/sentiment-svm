from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from ImportDataset import ImportDataset

import csv
import re 
import pandas as pd

class Preprocessor :

    def __init__(self, corpusPath):
         
        self.listStopword = set(stopwords.words('indonesian')) 
        factory = StemmerFactory() 
        self.stemmer = factory.create_stemmer() 
        
        self.importData = ImportDataset() 
        self.importData.readCsv(corpusPath)
        self.corpus = self.importData.getCorpus()
        
        self.kamus_katabaku = {} 
        katabaku = csv.reader(
                open("kata_baku.csv"), delimiter=";") 
       
        for row in katabaku : 
            self.kamus_katabaku[row[1]] = row[0]

        self.corpus['text'].dropna(inplace=True)

    def process(self): 
        komentar = [] 
        for index,row in enumerate(self.corpus['text']): #melakukan perulangan pada setiap baris komentar	
            
            kom = re.sub('[^A-Za-z]+',' ', row) # cleansing (regex) mengahpus tanda baca dan angka
            kom = kom.lower()# case folding (semua ke lower case)

            tokens = word_tokenize(kom) #tokenize, kalimat jadi array kata 

            removed = []
            for t in tokens:  #loop nyebutin setiap kata pada kalimat 
                    
                    try : 
                        t = ''.join(ch for ch, _ in itertools.groupby(t))
                        t = self.kamus_katabaku[t] # proses normalisasi, pemetaan kata non baku ke baku.
                    except :
                        pass  
                        
                    # negation handling (besok)
                        
                    if t not in self.listStopword and len(t) > 2: # jika kata itu gaada di listStopword berarti kata penting
                        removed.append(t)

            removed = " ".join(removed)
            katadasar = self.stemmer.stem(removed)
            #katadasar = katadasar.split(' ')
            katadasar = word_tokenize(katadasar)
            #komentar.append(removed) 
            print(katadasar)
            self.corpus.loc[index,'text_final'] = str(katadasar)

    def getCorpusText(self) :
        return self.corpus['text'] 

    def getCorpusTextFinal(self) : 
        return self.corpus['text_final'] 

    def getCorpusLabel(self): 
        return self.corpus['label']

    def getCorpusJson(self):
        return self.importData.toJSON()
