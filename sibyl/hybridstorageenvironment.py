from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
import numpy as np
from tf_agents.trajectories import time_step as timeStep
import time
import sys
NEW_DEBUG=True
N_spatial=5
class HybridStorageEnvironment(py_environment.PyEnvironment):
    def __init__(self, hybrid):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')  
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,8), dtype=np.float64, minimum=None, maximum=None, name='observation') 
        self._action_values = {0:'slow',1:'fast'}
        self._hybrid = hybrid
        self._current_perf=0
        self._hybrid._mapping_table.iloc[0:0]
        self._hybrid._mapping_table.drop( self._hybrid._mapping_table.index, inplace=True)
        self._total_perf=0
        self._hybrid._trace_index=0
        self._total_evicts=0
        self._total=0
        self._prev_action=1
        self._fast_action=0
        self._slow_action=0
        self._total_migrations=0
        self._total_invalid_page_fast=0
        self._total_invalid_page_slow=0
        self._total_access=0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._hybrid.reset()
        self._total_perf=0
        self._total_evicts=0
        self._total=0
        self._prev_action=1
        self._fast_action=0
        self._slow_action=0
        self._total_migrations=0
        self._total_invalid_page_fast=0
        self._total_invalid_page_slow=0
        self._total_access=0
        self._episode_ended = False
        self._hybrid._trace_index=0
        print([self._hybrid._state[0,:]])
        self._obs=np.append([self._hybrid._state[0,1:3]],[0,0,0,0,0,1])
        return timeStep.restart(np.array([self._obs], dtype=np.float64))

    def enter_env (self,action):
        current_state = np.zeros((1,3),dtype=np.float64)        
        current_state=np.array([self._hybrid._state[self._hybrid._trace_index,:]], dtype=np.float64)
        if str(float(self._hybrid._state[self._hybrid._trace_index,0])) in self._hybrid._mapping_table.index:
                self._prev_action=self._hybrid._mapping_table.at[str(self._hybrid._state[self._hybrid._trace_index,0]),'PrevAction']      
        else:
            self._prev_action=1
        #Perform placement
        self._current_perf=self._hybrid.placement(current_state,action)*1e3
        
    def _step(self, action): 
        if self._episode_ended:
            return self.reset()
        global VBA_prev
        self._obs = np.zeros((1,3),dtype=np.float64)    
        self._obs=np.array([self._hybrid._state[self._hybrid._trace_index,:]], dtype=np.float64)
        
        if(NEW_DEBUG==True):
            sys.stdout.write("> Observation vector: %s\n" % self._obs)
            sys.stdout.write("> Action: %d\n" % action)
        VBA_check=str(self._hybrid._state[self._hybrid._trace_index,0])

        if VBA_check in self._hybrid._metadata_table.index:
            self._hybrid._metadata_table.at[VBA_check,"accessCount1"] += 1
            curr_place=self._hybrid._metadata_table.at[VBA_check,"Device1"]     
            access_count=self._hybrid._metadata_table.at[VBA_check,"accessCount1"]
        else:
            access_count=0
            curr_place=1
        if VBA_check in self._hybrid._metadata_table.index:
            if(abs(self._hybrid._state[self._hybrid._trace_index,0]-VBA_prev)<=N_spatial):
                        self._hybrid._metadata_table.at[VBA_check,"spatialCount1"]+=1
            else:
                self._hybrid._metadata_table.at[VBA_check,"spatialCount1"]=0
            spatial_count=self._hybrid._metadata_table.at[VBA_check,"spatialCount1"]
        else:
            spatial_count=0

        if (self._hybrid._state[self._hybrid._trace_index,1]/4096>1):
            burst_count=self._hybrid._state[self._hybrid._trace_index,1]/4096
        else:
            burst_count=0

        self.enter_env(action)  

        prevReuse=self._hybrid._mapping_table.at[str(self._hybrid._state[self._hybrid._trace_index,0]),'ReuseDist']   
        self._hybrid._trace_index += 1       
        VBA_prev=self._hybrid._state[self._hybrid._trace_index-1,0]
        if(self._hybrid._trace_index==self._hybrid._trace_length):
            self._episode_ended = True

        self._total_access+=1
        if(action==1):
            self._fast_action+=1
        else:
            self._slow_action+=1
        if (self._hybrid._trace_index==self._hybrid._trace_length):
            self._episode_ended = True
        # if not (self._hybrid._trace_index==self._hybrid._trace_length):
        #     while(self._hybrid._state[self._hybrid._trace_index,2]==0):
        #         self.enter_env (action)
        #         if(NEW_DEBUG==True):
        #             sys.stdout.write("> Total time: {}\n" .format(self._total_perf/1e3))
              
        #         self._hybrid._trace_index += 1
        #         self._total_access+=1

        #         if(self._hybrid._trace_index==self._hybrid._trace_length):
        #             self._episode_ended = True
        #             break

        # else:
        #         self._episode_ended = True
        min_reuse=0
        max_reuse=self._hybrid._trace_length/2
        
     
        access_count = (access_count - min_reuse) / (max_reuse - min_reuse)
        self._obs=np.append([self._obs],[access_count])
        self._obs=np.append([self._obs],[spatial_count])
        self._obs=np.append([self._obs],[burst_count])
        self._obs=np.append([self._obs],[curr_place])
        filledPercent_fast =self._hybrid._devices.at["fastSSD","Filled"]/self._hybrid._devices.at["fastSSD","Capacity"]
        self._obs=np.append([self._obs],[filledPercent_fast])
        filledPercent_slow = self._hybrid._devices.at["slowSSD","Filled"]/self._hybrid._devices.at["slowSSD","Capacity"]
        self._obs=np.append([self._obs],[filledPercent_slow])

        self._total_evicts+=self._hybrid.numEvicts
        self._obs=self._obs[1:10]
       
       
        self._obs[0]=(self._obs[0]-self._hybrid.size_min)/(self._hybrid.size_max-self._hybrid.size_min) ## NORMALIZING PAGE SIZE
       
        self._total_perf+=self._current_perf


        if(NEW_DEBUG==True):
            array_sum = np.sum(self._obs)
            array_has_nan = np.isnan(array_sum)
            if(array_has_nan):
                sys.stderr.write("> Invalid observation vector\n")
           
            print("\t\t***************************************************************************************************")
        if(NEW_DEBUG==True):

            sys.stdout.write("> Size:{}, R/W:{}, Access count:{},\
                 Spatial count:{}, #pages:{}, #previous:{}, Filled_Fast:{},\
                  Filled_slow:{}\n".format(self._obs[0],self._obs[1],self._obs[2],self._obs[3],\
                        self._obs[4],self._obs[5],self._obs[6],self._obs[7]))
     

        self._obs=np.array([self._obs], dtype=np.float64)  
 
        migrations=self._hybrid._migration
        self._total_migrations+=migrations
        filledPercent_fast = self._hybrid._devices.at["fastSSD","Filled"]/self._hybrid._devices.at["fastSSD","Capacity"]
               
        filledPercent_slow = self._hybrid._devices.at["slowSSD","Filled"]/self._hybrid._devices.at["slowSSD","Capacity"]
        numMigrationsForSlowerSSD = self._hybrid._mapping_table['NumMigrationsSSD2'].sum()

        SlowSSDTotalWrites = (numMigrationsForSlowerSSD + self._hybrid._devices.at['slowSSD','WriteCount'])

        numMigrationsForFasterSSD = self._hybrid._mapping_table['NumMigrationsSSD1'].sum()
        FastSSDTotalWrites = (numMigrationsForFasterSSD + self._hybrid._devices.at['fastSSD','WriteCount'])
   
        if(NEW_DEBUG==True):
            
            print("\t\tMigrations",migrations)
            print("\t\tInvalid pages",self._hybrid._invalid_page_fast+self._hybrid._invalid_page_slow)
            print("\t\ttotal time:::",self._total_perf/1e3)
            print("\t\ttOtoal _current_perf:::",self._current_perf/1e3 )
            print("\t\tOnly write request:::", self._hybrid._reqLatency )
            print("\t\tOnly eviction request:::", self._hybrid._evictLatency)

            print("\t\tfilled fast={}, filled slow={}".format(filledPercent_fast,filledPercent_slow))

        reward_perf=(0.1/self._current_perf)-self._hybrid.numEvicts**0.09
        ############################################STEP REWARD#########################################################
        self._total_invalid_page_fast+=self._hybrid._invalid_page_fast
        self._total_invalid_page_slow+=self._hybrid._invalid_page_slow
        self._hybrid._migration=0
        self._hybrid._invalid_page_fast=0
        self._hybrid._invalid_page_slow=0
        self._hybrid._invalid_fast=0
        self._hybrid.numEvicts=0
        self._current_perf=0

        if (self._hybrid._trace_index==self._hybrid._trace_length):
            global slowDevice
            global fastDevice
            self._episode_ended = True
            numMigrationsForSlowerSSD = self._hybrid._mapping_table['NumMigrationsSSD2'].sum()
            SlowSSDTotalWrites = (numMigrationsForSlowerSSD + self._hybrid._devices.at['slowSSD','WriteCount'])
            numMigrationsForFasterSSD = self._hybrid._mapping_table['NumMigrationsSSD1'].sum()
            FastSSDTotalWrites = (numMigrationsForFasterSSD + self._hybrid._devices.at['fastSSD','WriteCount'])     
            SlowSSDReadCount = self._hybrid._devices.at['slowSSD','ReadCount']
            FastSSDReadCount = self._hybrid._devices.at['fastSSD','ReadCount']
            self._hybrid._trace_index=0
            self._total_evicts=0
            self._total_migrations=0
            self._hybrid._devices["Filled"] = 0
            self._hybrid._devices["WriteCount"] = 0
            self._hybrid._devices["ReadCount"] = 0
            self._hybrid._devices.at["fastSSD", "Filled"] = 0
            self._hybrid._devices.at["slowSSD", "Filled"] = 0
            self._hybrid._devices.at["fastSSD", "WriteCount"] = 0
            self._hybrid._devices.at["slowSSD", "WriteCount"] = 0
            print("\t############Action={}, Prev={}, Reward={}############".format(action,self._prev_action,reward_perf))
            return timeStep.termination(self._obs, reward=reward_perf)   

        if(NEW_DEBUG==True):
            print("\t############Action={}, Prev={}, Reward={}############".format(action,self._prev_action,reward_perf))
        return timeStep.transition(self._obs, reward=reward_perf, discount=1.0)


