import pandas as pd
import numpy as np
import time
import os
import sys
from ctypes import *

import logging
class HybridStorage():
    def __init__(self,application,driver_path):  
        self.application=application
        so_file = driver_path
        self.my_functions = CDLL(so_file)
        self.fastDevice = self.my_functions.openFastDevice()
        self.slowDevice = self.my_functions.openSlowDevice()
        self.gLBAFast = 1024 * 1024 * 1024
        self.gLBASlow = 1024 * 1024 * 1024
        cols=[0,1,2]
        col_name=[ 'col_offset1','col_bytes1','col_type']
        input_file=self.application
        trace_file=pd.read_csv( input_file,usecols=cols,  names=col_name, header=None 
                        ,dtype={'col_offset1':np.float32,'col_bytes1':np.float32}
                        )
        trace_file = trace_file[['col_offset1','col_bytes1','col_type']]
        trace_file['col_type'][trace_file['col_type'] == 'Write'] = 1 
        trace_file['col_type'][trace_file['col_type'] == 'Read'] = 0
        self._devices = pd.DataFrame(columns=["Device", "Capacity"])
        self._devices["Filled"] = 0
        self._devices["WriteCount"] = 0
        self._devices["ReadCount"] = 0
        self._devices.set_index('Device', inplace=True)
        self._devices.at['fastSSD', 'Capacity'] = (2 * 1024 * 1024 * 1024) 
        self._devices.at["slowSSD", "Capacity"] = (1.76 * 1024 * 1024 * 1024 * 1024) 
        self._devices.at["fastSSD", "Filled"] = 0
        self._devices.at["slowSSD", "Filled"] = 0
        self._devices.at["fastSSD", "WriteCount"] = 0
        self._devices.at["slowSSD", "WriteCount"] = 0
        self._devices.at["fastSSD", "ReadCount"] = 0
        self._devices.at["slowSSD", "ReadCount"] = 0
        self.SSDFEvictionThreshold = 90
        self._trace_index = 0
        self.metadata_time=0
        self.qrator_proc_time=0
        self.exec_time=0
        self.numEvicts = 0
        self._current_fast_capacity=0
        self._fast_capacity=50
        self._final_latency=0
        self._migration=0
        self._reqLatency=0
        self._evictLatency=0
        self._invalid_page_fast=0
        self._invalid_page_slow=0
        self._invalid_fast=0
        self.size_max=trace_file['col_bytes1'].max()
        self.size_min=trace_file['col_bytes1'].min()
        self._mapping_table = pd.DataFrame(columns=["Size", "ReadWrite", "Device","LBA", "LatestWriteCount", "LatestReadCount",\
         "TotalWrites", "TotalReads","NumMigrationsSSD1", "NumMigrationsSSD2", "PrevAction","CurAction","ReuseDist"
         ])
        self._metadata_table=pd.DataFrame(columns=[ "accessCount", "accessInterval","Device"])
        self.all_addr_freq={}
        self._trace_length=len(trace_file.index)
        self._trace_shape=trace_file.shape
        self._state = np.zeros((self._trace_shape),dtype=np.float64)   
        self._df=trace_file.values
        self._state=self._df


    def reset(self): 
        self._state = np.zeros((self._trace_shape),dtype=np.float64)  
        self._state=self._df
        self.numEvicts = 0
        self._trace_index = 0
        self._current_fast_capacity=0
        self._final_latency=0
        self._reqLatency=0
        self._evictLatency=0
        self._invalid_page_fast=0
        self._invalid_page_slow=0
        self._mapping_table.iloc[0:0]
        self._metadata_table.iloc[0:0]
        self._invalid_fast=0
        self._mapping_table.drop( self._mapping_table.index, inplace=True)
        self._metadata_table.drop( self._metadata_table.index, inplace=True)
        self._devices = pd.DataFrame(columns=["Device", "Capacity"])
        self._devices["Filled"] = 0
        self._devices["WriteCount"] = 0
        self._devices["ReadCount"] = 0
        self._devices.set_index('Device', inplace=True)
        self._devices.at['fastSSD', 'Capacity'] = (2 * 1024 * 1024 * 1024) 
        self._devices.at["slowSSD", "Capacity"] = (1.76 * 1024 * 1024 * 1024 * 1024) #1.76 TB is the size of the slow SSD
        self._devices.at["fastSSD", "Filled"] = 0
        self._devices.at["slowSSD", "Filled"] = 0
        self._devices.at["fastSSD", "WriteCount"] = 0
        self._devices.at["slowSSD", "WriteCount"] = 0
        self._devices.at["fastSSD", "ReadCount"] = 0
        self._devices.at["slowSSD", "ReadCount"] = 0
    
    
    
    def promote(self,request, readFlag): # To promote a request from slow SSD to fast SSD

        latency = 0
        start = 0
        end = 0
        VBA = str(request[0])
        oldLBA = self._mapping_table.at[VBA, "LBA"]
        newSize = request[1]
        finalSize = 0
        totalSize = self._mapping_table.at[VBA, "Size"]
        
        if newSize > totalSize :
            finalSize = newSize
        else:
            finalSize = totalSize
            
        newLBA = int(self.gLBAFast)
        self.gLBAFast += finalSize
    
        self._metadata_table.at[VBA, "Device"] = 1 # Promote the page from slow SSD to fast SSD
        self._mapping_table.at[VBA, "Device"] = "fastSSD" # Promote the page from slow SSD to fast SSD
        self._mapping_table.at[VBA, "LBA"] = newLBA
        self._mapping_table.at[VBA, "NumMigrationsSSD1"] += 1
        totalSize = self._mapping_table.at[VBA, "Size"]
        self._devices.at["slowSSD", "Filled"] -= totalSize
        self._devices.at["slowSSD", "ReadCount"] += 1
        
        if readFlag == False:
            currSize = int(request[1])
            
            if currSize < totalSize:
            
                self._devices.at["fastSSD", "Filled"] += totalSize
                
                start = time.perf_counter()
                self.my_functions.sibyl_read(self.slowDevice, oldLBA+currSize, totalSize-currSize)
                end = time.perf_counter()           
                latency += (end - start)
                
                start = time.perf_counter()
                self.my_functions.sibyl_write(self.fastDevice, newLBA, totalSize)
                end = time.perf_counter()                    
                latency += (end - start)
                
            else:
                self._mapping_table.at[VBA, "Size"] = currSize
                self._devices.at["fastSSD", "Filled"] += currSize
                start = time.perf_counter()
                self.my_functions.sibyl_write(self.fastDevice, newLBA, currSize)
                end = time.perf_counter()            
                latency += (end - start)
                
        else: # ReadFlag is True
            self._devices.at["fastSSD", "Filled"] += totalSize
        
            start = time.perf_counter()
            self.my_functions.sibyl_read(self.slowDevice, oldLBA, totalSize)
            end = time.perf_counter()        
            latency += (end - start)
            
            start = time.perf_counter()
            self.my_functions.sibyl_write(self.fastDevice, newLBA, totalSize)
            end = time.perf_counter()                
            latency += (end - start)
        
        #print("Promote Latency ", latency)
    
        return latency
    def evict(self,request, readFlag): # To evict a request from fast SSD to slow SSD
        # global gLBASlow  
        latency = 0
        start = 0
        end = 0
        VBA = str(request[0])
        oldLBA = self._mapping_table.at[VBA, "LBA"]
        newSize = request[1]
        finalSize = 0
        totalSize = self._mapping_table.at[VBA, "Size"]
        
        if newSize > totalSize :
            finalSize = newSize
        else:
            finalSize = totalSize
            
        newLBA = int(self.gLBASlow)
        self.gLBASlow += finalSize
    
        #if VBA in self._mapping_table.index:
        self._metadata_table.at[VBA, "Device"] = 0 # Evict the page from slow SSD to fast SSD
        self._mapping_table.at[VBA, "Device"] = "slowSSD" # Evict the page from slow SSD to fast SSD
        self._mapping_table.at[VBA, "LBA"] = newLBA
        self._mapping_table.at[VBA, "NumMigrationsSSD2"] += 1
        totalSize = self._mapping_table.at[VBA, "Size"]
        self._devices.at["fastSSD", "Filled"] -= totalSize
        self._devices.at["fastSSD", "ReadCount"] += 1
        
        if readFlag == False:
            currSize = int(request[1])
            
            if currSize < totalSize:
            
                self._devices.at["slowSSD", "Filled"] += totalSize
                
                start = time.perf_counter()
                self.my_functions.sibyl_read(self.fastDevice, oldLBA+currSize, totalSize-currSize)
                end = time.perf_counter()           
                latency += (end - start)
                
                start = time.perf_counter()
                self.my_functions.sibyl_write(self.slowDevice, newLBA, totalSize)
                end = time.perf_counter()                    
                latency += (end - start)
                
            else:
                self._mapping_table.at[VBA, "Size"] = currSize
                self._devices.at["slowSSD", "Filled"] += currSize
                start = time.perf_counter()
                self.my_functions.sibyl_write(self.slowDevice, newLBA, currSize)
                end = time.perf_counter()            
                latency += (end - start)
                
      
    
        return latency

    def urgentEviction(self,curReqSize):
        start_time = time.perf_counter()
        start = 0
        end = 0
        global NumEvictions
        self.numEvicts = 0
        latency = 0
        filledPercent = 0
        self._mapping_table = self._mapping_table.sort_values(['Device', 'ReuseDist'], ascending=[True, False])
        sizeUpto = 0
        for ind in self._mapping_table.index:
            if self._mapping_table["Device"][ind] == "fastSSD":
                self.numEvicts += 1
                self._devices.at["fastSSD", "ReadCount"] += 1
                i = 0
                VBA = ind
                currSize = self._mapping_table.at[VBA, "Size"]
                newLBA = int(self.gLBASlow)
                self.gLBASlow += currSize
                sizeUpto += currSize           
                oldLBA = self._mapping_table.at[VBA, "LBA"]
                self._mapping_table.at[VBA, "Device"] = "slowSSD"
                self._mapping_table.at[VBA, "LBA"] = newLBA
                self._devices.at["fastSSD", "Filled"] -= currSize
                self._devices.at["slowSSD", "Filled"] += currSize
                self._mapping_table.at[VBA, "NumMigrationsSSD2"] += 1
                self._metadata_table.at[VBA,"Device"]= 0
                start = time.perf_counter()
                self.my_functions.sibyl_read(self.fastDevice, oldLBA, currSize)
                end = time.perf_counter()
                latency += (end - start) 
                start = time.perf_counter()
                self.my_functions.sibyl_write(self.slowDevice, newLBA, currSize)
                end = time.perf_counter()           
                latency += (end - start) 
                if (sizeUpto >= curReqSize):
                    break
        return latency


    def read(self,obs,memory_type):
        latency = 0
        deviceName = ""
        request=[]
        request.append(str(obs[0][0]))
        request.append(int(obs[0][1]))
        request.append(int(obs[0][2]))
        VBA = str(request[0])
        if VBA in self._mapping_table.index:
            
            if(memory_type==1):
                    deviceName = "fastSSD"
            else:
                    deviceName = "slowSSD"
            self._mapping_table.at[VBA,"TotalReads"] += 1
            self._mapping_table.at[VBA,"ReadWrite"] = 'Read'
            self._mapping_table.at[VBA,"ReuseDist"] = 0
            self._devices.at[deviceName, "ReadCount"] += 1 
            self._mapping_table.at[VBA,"LatestReadCount"] += 1
            LBA = self._mapping_table.at[VBA,"LBA"] 
            #Calculate latency
            #It is a sequential read, multiply sequential read latency by the number of chunks read
            
            #####################START
            dname = self._mapping_table.at[request[0],"Device"] #current device

            if (dname == "slowSSD") and (deviceName == "fastSSD"):
                logging.debug("\t\t\t[READ]****PROMOTE****")
                latency += self.promote(request, False)

            elif (dname == "fastSSD") and (deviceName == "slowSSD"):
                logging.debug("\t\t\t[READ] ****EVICT****")
                latency += self.evict(request, False)
                
            else:
                logging.debug("\t\t\t[READ] ****NO MOVE****")
            #####################END
            readSize = int(request[1])
            if deviceName == "slowSSD":
                start = time.perf_counter()
                self.my_functions.sibyl_read(self.slowDevice, LBA, readSize)
                end = time.perf_counter()
            
            if deviceName == "fastSSD":
                start = time.perf_counter()
                self.my_functions.sibyl_read(self.fastDevice, LBA, readSize)
                end = time.perf_counter()
            latency += (end - start) 
     
        return latency

    def write(self,obs,memory_type):
        start = 0
        end = 0
        latency = 0
        createMapping = True
        newRequest = False    
        request_metadata=[]
        request=[] 
        request.append(str(obs[0][0]))
        request.append(int(obs[0][1]))
        request.append(int(obs[0][2]))

        deviceName = ""
        dname = ""  
        self._evictLatency = 0
        VBA = str(request[0])  
        newSize = request[1]
        sizeToMove = newSize  
        not_newRequest=False
        meta_access=0
        meta_spatial=0
        meta_burst=0
        if request[0] in self._mapping_table.index: #checking if the table already exists
                dname = self._mapping_table.at[request[0],"Device"] #current device
                currSize = self._mapping_table.at[VBA,"Size"]
                oldLBA = self._mapping_table.at[VBA, "LBA"]
                writeCounter = self._mapping_table.at[request[0],"LatestWriteCount"]
                writeCounter += 1
                readCounter = self._mapping_table.at[request[0],"LatestReadCount"]
                totalWriteCounter = self._mapping_table.at[request[0],"TotalWrites"] + 1
                totalReadCounter = self._mapping_table.at[request[0],"TotalReads"]
                numMigrations1 = self._mapping_table.at[request[0],"NumMigrationsSSD1"]
                numMigrations2 = self._mapping_table.at[request[0],"NumMigrationsSSD2"]
                prev_act=self._mapping_table.at[request[0],'CurAction']
                self._mapping_table.at[request[0],'PrevAction']=prev_act
                self._mapping_table.at[request[0],'CurAction']=memory_type
                reuse = 0
                if newSize > currSize:
                    sizeToMove = newSize
                else:
                    sizeToMove = currSize  
                if(memory_type==1):
                    deviceName = "fastSSD"
                else:
                    deviceName = "slowSSD"
                if(deviceName == "fastSSD"):
                    if newSize > currSize:
                        extraSize = newSize-currSize
                        filledPercent = 100*(self._devices.at["fastSSD","Filled"] + extraSize)/self._devices.at["fastSSD","Capacity"]
                        if filledPercent > self.SSDFEvictionThreshold:
                            self._evictLatency = self.urgentEviction(extraSize)
                    
                if dname!=deviceName and dname!='':
                        if(dname=='fastSSD'):
                            self._invalid_page_fast=1
                        if(dname=="slowSSD"):
                            self._invalid_page_slow=1  
                #Check if the new size is greater than currSize then creat a new mapping
                if newSize > currSize:
                    self._devices.at[dname, "Filled"] -= currSize
                    prev_reuse_dist=self._mapping_table.at[VBA, "ReuseDist"] 
                    self._mapping_table.drop(VBA, inplace=True)
                    not_newRequest=True
                else: #for currSize <= new Size; Just update the LBAs corresponding to the size and other metadata
                    createMapping = False # To not update the mapping table
                    self._metadata_table.at[VBA,"Device"] = memory_type
                    self._mapping_table.at[VBA,"LatestWriteCount"] = writeCounter
                    self._mapping_table.at[VBA,"TotalWrites"] = totalWriteCounter
                    self._mapping_table.at[VBA,"Device"] = deviceName
                    self._mapping_table.at[VBA, "ReuseDist"] = 0
        #If it not yet in the mapping table, write to SSD1 unless SSD1 is already filled beyond threshold
        else:
            newRequest = True             
            if(memory_type==1):
                    deviceName = "fastSSD"
            else:
                    deviceName = "slowSSD"
            if(deviceName=="fastSSD"):
                filledPercent = 100*(self._devices.at["fastSSD","Filled"]+newSize)/self._devices.at["fastSSD","Capacity"]
                if filledPercent > self.SSDFEvictionThreshold:
                    self._evictLatency = self.urgentEviction(newSize)
            writeCounter = 1
            readCounter = 0
            totalWriteCounter = 1
            totalReadCounter = 0
            numMigrations1 = 0
            numMigrations2 = 0
            reuse = 0
            meta_access=0
            meta_spatial=0
        #Calculate latency
        latency = 0
        if createMapping == True: #Create new mapping
            if deviceName == "slowSSD":
                LBA = int(self.gLBASlow)
                self.gLBASlow += sizeToMove
            
            else: # Fast SSD
                LBA = int(self.gLBAFast)
                self.gLBAFast += sizeToMove

            request_metadata.append(meta_access)
            request_metadata.append(meta_spatial)
            request_metadata.append(memory_type)
            request.append(deviceName)
            request.append(LBA)
            request.append(writeCounter)
            request.append(readCounter)
            request.append(totalWriteCounter)
            request.append(totalReadCounter)
            request.append(numMigrations1)
            request.append(numMigrations2)
            request.append(1) #beginning the action is 1
            request.append(memory_type)
            request.append(reuse)
            self._devices.at[deviceName, "Filled"] += sizeToMove
            self._mapping_table.at[VBA] = request[1:]
            self._mapping_table.at[VBA, "Size"] = sizeToMove
            self._metadata_table.at[VBA] = request_metadata[0:]

        else: # No new mapping table entry is required
            LBA =  self._mapping_table.at[VBA, "LBA"]
        self._mapping_table.at[VBA,"ReadWrite"] = 'Write'
        self._devices.at[deviceName, "WriteCount"] += 1
        #Calculate latency
        if deviceName == "fastSSD":
            start = time.perf_counter()
            self.my_functions.sibyl_write(self.fastDevice, LBA, newSize)
            end = time.perf_counter()                 
            latency = (end - start)
        else:
            start = time.perf_counter()
            self.my_functions.sibyl_write(self.slowDevice, LBA, newSize)
            end = time.perf_counter()                 
            latency = (end - start) 
        return latency

    
    def placement(self, obs, action):
        self._mapping_table.ReuseDist += 1
        logging.debug("\t> Request type: {} (1: WRITE, 0: READ)".format(int(obs[0][2])))
      
        if(int(obs[0][2])==1):
            self._reqLatency= self.write(obs, action)
        else:
            self._reqLatency= self.read(obs, action)
        self._final_latency=self._reqLatency    
        return self._final_latency

