#encoding:utf-8
import sys
"""
"""
f=open('top40000_dnn.vector')
datas=f.readlines()
id_dict=dict()

index=0
for line in datas:
	index=index+1
	line=line.strip().split(' ')
	id_dict[line[0]]=str(index)

for item in sys.stdin:
	try:
		src_item=item.strip()
		item=item.strip()
		words=item
		words_list=words.split(' ')
		id_list=list()
		for item in words_list:
			if item in id_dict:
				id=id_dict[item]
				id_list.append(id)
		print src_item+"\t"+" ".join(id_list)
	except:
		pass
