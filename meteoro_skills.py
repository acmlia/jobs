#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:48:36 2019

@author: dvdgmf
"""


class MeteorologicalDiagnosis:
    
    def __init__(self, obs, pred):
        self.tptn = self.get_TPTN(obs, pred)
        self.tn = self.tptn.count('TN')
        self.tp = self.tptn.count('TP')
        self.fn = self.tptn.count('FN')
        self.fp = self.tptn.count('FP')
        
    @staticmethod
    def get_TPTN(obs, pred):
        if len(obs) == len(pred):
            if all(O in [0, 1] for O in obs) and all(P in [0, 1] for P in pred):
                output_list = []
                test = {
                    (0, 0): "TN",
                    (1, 1): "TP",
                    (0, 1): "FN",
                    (1, 0): "FP"
                }
                for O, P in zip(obs, pred):
                    output_list.append(test.get((O, P)))
                return output_list
            else:
                print('Invalid values present in Y-Observed and/or Y-Predicted.'
                      'Only binary values 0 and 1 are supported!')
        else:
            print('Invalid list size. Y-Observed and Y-Predicted must match!')

    def metrics(self):
        val_accuracy = self.tryMetrics(self.accuracy)
        val_bias = self.tryMetrics(self.bias)
        val_pod = self.tryMetrics(self.pod)
        val_pofd = self.tryMetrics(self.pofd)
        val_far = self.tryMetrics(self.far)
        val_csi = self.tryMetrics(self.csi) 
        val_ph = self.tryMetrics(self.ph) 
        val_ets = self.tryMetrics(self.ets) 
        val_hss = self.tryMetrics(self.hss) 
        val_hkd = self.tryMetrics(self.hkd)
        return val_accuracy, val_bias, val_pod, val_pofd, val_far, val_csi, val_ph, val_ets, val_hss, val_hkd

    def tryMetrics(self, fx):
        tptn = self.tptn
        tn = self.tn
        tp = self.tp
        fn = self.fn
        fp = self.fp
        try:
            result = fx(tptn, tn, tp, fn, fp)
            return result
        except ZeroDivisionError as e:
            pass
            print('ZeroDivisionError:', e)
            return None
        except TypeError as e:
            pass
            print('TypeError:', e)
            return None

    def accuracy(self, tptn,tn,tp,fn,fp):
        return (tp + tn) / len(tptn)
    
    def bias(self, tptn,tn,tp,fn,fp):
        return (tp + fp) / (tp + fn)
    
    def pod(self, tptn,tn,tp,fn,fp):
        return  tp / (tp + fn)
    
    def pofd(self, tptn,tn,tp,fn,fp):
        return fp / (fp + tn)
    
    def far(self, tptn,tn,tp,fn,fp):
        return  fp / (tp + fp)
    
    def csi(self, tptn,tn,tp,fn,fp):
        return tp / (tp + fp + fn)
    
    def ph(self, tptn,tn,tp,fn,fp):
        return ((tp + tn) * (tp + fp)) / (tp + tn + fp + fn)
        
    def ets(self, tptn,tn,tp,fn,fp):
        val_ph = self.tryMetrics(self.ph)
        return (tp - val_ph) / (tp + fp + fn - val_ph)

    def hss(self, tptn,tn,tp,fn,fp):
        return ((tp * tn) - (fp * fn)) / ((((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))) / 2)
    
    def hkd(self, tptn,tn,tp,fn,fp):
        val_pod = self.tryMetrics(self.pod)
        val_pofd = self.tryMetrics(self.pofd)
        return val_pod - val_pofd
    
# ---------------------------------------------

if __name__ == '__main__':
    obs =   [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
    pred =  [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    mozao_tools = MeteorologicalDiagnosis(obs, pred)
    val_accuracy, val_bias, val_pod, val_pofd, val_far, val_csi, val_ph, val_ets, val_hss, val_hkd = mozao_tools.metrics()
    print('metrics:\n accuracy: {}\n bias: {}\n pod: {}\n pofd: {}\n far: {}'
          '\n csi: {}\n ph: {}\n ets: {}\n hss: {}\n hkd: {}\n'.format(val_accuracy,
                   val_bias, val_pod, val_pofd, val_far, 
                   val_csi, val_ph, val_ets, val_hss, val_hkd))
