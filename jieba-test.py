import jieba
import jieba.analyse
# https://blog.csdn.net/ZJRN1027/article/details/103513861

seg_str="员工简川力家庭地址在南山区粤美特大厦，邮箱地址为是jianchuanli@qianxin.com"

print("/".join(jieba.lcut(seg_str)))    # 精简模式，返回一个列表类型的结果
print("/".join(jieba.lcut(seg_str, cut_all=True)))      # 全模式，使用 'cut_all=True' 指定 
print("/".join(jieba.lcut_for_search(seg_str)))     # 搜索引擎模式


import jieba.analyse
s = seg_str #"此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
for t, w in jieba.analyse.textrank(s, withWeight=True):
	print('%s, %s'% (t, w))