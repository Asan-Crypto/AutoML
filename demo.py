import os
import sys
import requests
import base64
import json
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import swat
from swat.render import render_html
from swat import *
import numpy as np
import pandas as pd
from sasctl import register_model, Session
import datetime
import time

class Model:
    def __init__(self, task = 'default', procId = '', init = False, WithCAS = False):
        self.task = task
        self.protocol = 'http'
        self.host = 'sasdemo.sas.com'
        self.username = 'sasdemo'
        self.password = 'Orion123'
        self.authUri = '/SASLogon/oauth/token'
        casport=5570

        r = requests.post(
            'http://'+ self.host + self.authUri,
            params={
              'grant_type': 'password',
              'username': self.username,
              'password': self.password
            },
            headers = {
                'Authorization': '',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        )
        self.token = json.loads(r.text)['access_token']
        
        if WithCAS:
            self.sess = CAS("sasdemo.sas.com", casport, username=self.username, password=self.password, protocol = '')
            self.sess.sessionProp.setSessOpt(caslib = '')

        if task == "AutoML":
            self.raw_inputs = ['CA_Nav_Bar_Clicks', 'PAGETYPE_Clearance', 'PAGETYPE_Homepage',
           'PAGETYPE_SearchResults', 'PAGETYPE_ProductListPage',
           'PAGETYPE_Product_Page', 'Basket_Quantity', 'Visit_Before_Work',
           'Visit_After_Work', 'Visit_Evening', 'Pages_Seen', 'Pages_gt1',
           'PRODUCT_VIEWS_FINAL', 'PRODUCT_VIEWS_gt1', 'Clearance']
        
        if init:
            self.setProjectProps()
            if(self.flag == 1):
                self.RandomModel()

        else:
            self.wfProcessId = procId
            
    def loadWorkflowProps(self):
        headers = {
            'Authorization': 'Bearer ' + self.token
        }
 #       url = str(self.protocol)+"://"+str(self.host)+"/modelManagement/workflowAssociations"
 #       workflowMeta = requests.get(url, headers=headers).json()
 #       processCount = workflowMeta["count"]

 #       url = str(self.protocol)+"://"+str(self.host)+"/modelManagement/workflowAssociations?limit=100"
        url = "http://sasdemo.sas.com/workflowAssociations?start=20"
        items = requests.get(url, headers=headers).json()["items"]
        workflowMeta = sorted(items, key=lambda k: parser.parse(k['']), reverse=True)[0]
        
        #if "processId" in workflowMeta:
        #    self.log("Loaded workflow process id and project id.")

        return (workflowMeta["processId"], workflowMeta["solutionObjectId"])
    
    def setProjectProps(self):
        self.wfProcessId, projectId = self.loadWorkflowProps()
        if(len(projectId) > 5):
            self.flag = 1
        else:
            self.flag = 0
        self.setWorkflowVar("procId", self.wfProcessId)

    
    
    def getWorkflowVar(self, name, getSchema=False):
        headers={
            'Accept-Language': 'en_us',
            'Authorization': 'Bearer ' + self.token
        }        
        url = str(self.protocol)+"://"+str(self.host)+""+str(self.wfProcessId)+""+str(name)
        schema = requests.get(url, headers=headers).json()

        if getSchema:
            return schema
        
        return schema.get('value', None)
    
    def setWorkflowVar(self, name, value):
        headers={
            'Accept-Language': 'en_us',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.token
        }
        schema = self.getWorkflowVar(name, getSchema=True)
        for key in ['links', 'localizedName', 'description']:
            if key in schema:
                del schema[key]
        schema['value'] = value
        
        url = str(self.protocol)+"://"+str(self.host)+"/workflow/processes/"+str(self.wfProcessId)+"/variables/"+str(name)
        requests.put(url, headers=headers, json=schema).json()
        
    def RandomModel(self):
        table = self.sess.CASTable("DF_2018_07_28").head(len(self.sess.CASTable("DF_2018_07_28")))
        table = table.drop({'complete_order'}, axis = 1)
        nums = np.random.choice([0, 1], size=len(table), p=[.5, .5])
        table['target'] = pd.Series(nums)
        X = table.drop({'Var1', 'session_id', 'session_dt', 'browser_nm', 'region_nm',
            'metro_cd', 'device_nm', 'device_type_nm', 'platform_desc',
            'profile_nm1', 'screen_size_txt', 'active_sec_spent_in_sessn_cnt',
            'seconds_spent_in_session_cnt', 'session_start_dttm', 'Hour',
            'GOAL_Order_Complete', 'GOAL_Success_Card', 'GOAL_Failure_Card',
            'CA_Product_Viewer', 'CA_Selected_Predicted_Search_Ter', 'CA_Read_Review', 'CA_HP_Carousel_Scroll',
            'CA_HP_Carousel_Click', 'PAGETYPE_Account',
            'PAGETYPE_Checkout', 'PAGETYPE_Other', 'Add_to_Basket_TOTAL',
            'Add_to_Basket_CNT', 'Basket_Value',
            'Existing_Basket_on_Session', 'Look_at_Stock', 'Saw_SoldOut_sum',
            'Saw_SoldOut_cnt', 'SoldOut_pct', 'Product_views',
            'Product_views_Unique', 'Product_Views_cnt', 'Basket_Removals',
            'Basket_Removals_cnt', 'Visit_Lunch', 'Complete_Order_cnt', 'CA_Read_Review_cnt', 'abandoned_order', 'target'}, axis = 1)
        y = table.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
        clf = RandomForestClassifier(max_depth=10)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        print(clf.score(X_test, y_test))
        day = self.getWorkflowVar('Day', False)
        month = self.getWorkflowVar('Month', False)
        if str(month) != "None" and str(day) != "None":
          with Session(hostname = "sasdemo.sas.com", username=self.username, password = self.password, verify_ssl=False):
              model = register_model(clf, 'randomModel' + str(day) + str(month), 'Demo Project', input = X_train, force=True)
            
          self.setWorkflowVar('Model_Name', 'randomModel' + str(day) + str(month))
          model_id = ''
          
          headers = {
              'Authorization': 'Bearer ' + self.token,
              'Content-Type': 'application/json'
          }
          url = str(self.protocol)+"://"+str(self.host)+""
          models = requests.get(url, headers=headers).json()['items']
  
          model_id = sorted(models, key=lambda k: parser.parse(k['']), reverse=True)[0]
          
          self.setWorkflowVar('modelid', model_id[''])
          self.setWorkflowVar('PublishDestination', '')
            
    def LoadTable(self):
        Day = self.getWorkflowVar("Day", False)
        Month = self.getWorkflowVar("Month", False)
  
        if Day == 31:
            self.setWorkflowVar('Month', Month + 1)
            self.setWorkflowVar('Day', 1)
            
        else:
            self.setWorkflowVar('Day', Day + 1)
            
    def EnoughResp(self):
        Day = self.getWorkflowVar("Day", False)
        Month = self.getWorkflowVar("Month", False)
        data_name = self.getWorkflowVar("data_name", False)
        
        TableName = ""
        
        
        if Day < 10:
            TableName = "DF_2018_0" + str(Month) + "_0" + str(Day)
            
        else:
            TableName = "DF_2018_0" + str(Month) + "_" + str(Day)
            
            
        new_data_len = len(self.sess.CASTable(TableName))
        
        
        if new_data_len - len(self.sess.CASTable(data_name)) > 100:
            self.setWorkflowVar("EnoughResponces", True)
            self.setWorkflowVar("data_name", TableName)
            
        else:
            self.setWorkflowVar("EnoughResponces", False)
            
    def AutoML(self):
        #data_name = self.getWorkflowVar("data_name", False)

        self.sess.loadactionset(actionset="dataSciencePilot")
        r = self.sess.dataSciencePilot.dsAutoMl(
            table = {
                'name': 'TRAIN',
            },
            inputs = self.raw_inputs,
            target = 'complete_order',
            modelTypes = [
                'forest',
                'gradboost'
            ],
            kFolds = 2,
            objective = 'AUC',
            sampleSize = 1, # maximum number of pipelines to sample
            transformationPolicy = {
                'missing': True,
                'cardinality': True,
                'entropy': True,
                'iqv': True,
                'skewness': True,
                'kurtosis': True,
                'outlier': True
            },
            transformationOut = {
                'name': 'transformation',
                'replace': True
            },
            featureOut = {
                'name': 'feature_list',
                'replace': True
            },
            pipelineOut = {
                'name': 'pipeline',
                'replace': True
            },
            saveState = {
                'replace': True,
                'modelNamePrefix': 'dsaml',
                'topK': 2
            },
            seed = 1234
        )  
        
    def AssessModel(self):
        self.sess.astore.score(
            rstore = 'DSAML_FM_',
            table = {
                'name': 'DF_2018_07_30',
            },
            casOut = {
                'name': 'tmp',
                'replace': True
            },
            copyVars = ['complete_order']
        )
        
        _ = self.sess.astore.score(
            rstore = self.best_pipeline,
            table = 'tmp',
            casOut = {
                'name': 'dsaml_scored',
                'replace': True
            },
            copyVars = ['complete_order']
        )
        
        ad = self.sess.percentile.assess(
            table = 'dsaml_scored',
            inputs = [{'name': 'P_complete_order1'}],
            response = 'complete_order',
            event = '1',
            cutStep = 1 / len(self.sess.CASTable('dsaml_scored'))
        )
        print('AUC ROC for the best pipeline = %f' % ad['ROCInfo']['C'][0])

        
        
    def RegisterChampionModel(self):

        self.best_pipeline = 'dsaml_' + self.sess.CASTable('pipeline') \
            .to_frame(fetchvars = ['MLType', 'Objective']) \
            .sort_values(by = 'Objective', ascending = False)['MLType'][0] \
            + '_1'
    
        #self.AssessModel()
        day = self.getWorkflowVar('Day', False)
        month = self.getWorkflowVar('Month', False)
        with Session(hostname="rusid1.rus.sas.com", username=self.username, password = self.password, verify_ssl=False):
          model = register_model(self.sess.CASTable(self.best_pipeline), 'AutoML' + str(day) + str(month), 'Demo Project', force=True)

        self.setWorkflowVar('Model_Name', 'AutoML' + str(day) + str(month))
        model_id = ''
        
        headers = {
            'Authorization': 'Bearer ' + self.token,
            'Content-Type': 'application/json'
        }
        url = str(self.protocol)+"://"+str(self.host)+"/modelRepository/models?limit=100"
        models = requests.get(url, headers=headers).json()['items']

        model_id = sorted(models, key=lambda k: parser.parse(k['creationTimeStamp']), reverse=True)[0]
        
        self.setWorkflowVar('modelid', model_id['id'])
        self.setWorkflowVar("EnoughResponces", False)
        
    def Print(self):
        Day = self.getWorkflowVar("Day", False)
        Month = self.getWorkflowVar("Month", False)
        enough = self.getWorkflowVar("EnoughResponces", False)
        proc = self.getWorkflowVar("procId", False)
        data_name = self.getWorkflowVar("data_name", False)
        print(enough)
        print(proc)
        print(Day)        
        print(data_name)
        

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enough arguments! Usage: python "+str(sys.argv[0])+" [procId] <command>")
        exit()
    else:
        if sys.argv[1] == 'init':
            model = Model(task='init', init=True, WithCAS = True)
        
        if sys.argv[2] == 'AutoML':
            model = Model(task = 'AutoML', procId = sys.argv[1], WithCAS = True)
            model.AutoML()
            model.RegisterChampionModel()
        
        elif sys.argv[2] == 'UpdateResp':
            model = Model(procId = sys.argv[1], WithCAS=True)
            model.LoadTable()
                
        elif sys.argv[2] == 'EnoughResp':
            model = Model(procId = sys.argv[1], WithCAS=True)
            model.EnoughResp()
