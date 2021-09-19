# -*- coding: utf-8 -*-
import os
os.environ['TORCH_HOME']='E:\\pytorch'
import jieba
import jieba.posseg as jp


print(jp.lcut("员工简川力家庭地址在南山区粤美特大厦，邮箱地址为是jianchuanli@qianxin.com"))
