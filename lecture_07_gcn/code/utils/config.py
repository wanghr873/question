#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2020/2/9 2:45 下午 
# @Author  : Roger 
# @Version : V 0.1
# @Email   : 550997728@qq.com
# @File    : config.py
import os
import pathlib

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

bert_vocab_path = os.path.join(root, 'model', 'bert_model_chinese', 'vocab.txt')

baidu95_dataset_path = os.path.join(root, 'data', 'baidu_95.csv')

dataset_output_dir = os.path.join(root, 'data', 'dataset')
