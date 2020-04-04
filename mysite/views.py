from django.http import HttpResponse
import datetime
from django.template import Template, Context, RequestContext
import django.contrib.staticfiles
from django.shortcuts import render, render_to_response
from django.views.decorators.csrf import csrf_protect
import chardet
import time
from mysite.tasks import get_ner_result
    
    
    
def ner(request):
        '''
        fp = open('mysite/static/ner.html')#relative to manage.py

        t = Template(fp.read())
        fp.close()

        html = t.render(Context(request))   #replaced,request!!
        return HttpResponse(html)'''
        fp = open('mysite/static/chinese.html')#relative to manage.py
		# 先读取文件内容
        #file_content = "".join(fp.readlines())
	    # 只要不是 utf-8编码，一率用 gb2312 解码
        #if chardet.detect(file_content)["encoding"] != "utf-8":
            #file_content = file_content.decode("gb2312")

        t = Template(fp.read())
        #t = Template(file_content)
        fp.close()
        
        html = t.render(RequestContext(request))   #replaced  and can only be found this way..
        return HttpResponse(html)


#@csrf_protect
def result(request):
    now = datetime.datetime.now()
    name= "friend"
    email = 'the other side of the world'
    fp = open('mysite/static/result.html')
    t = Template(fp.read())
    fp.close()
    if request.method == 'POST':
        txt = request.POST['text']
        model = request.POST['model']
        '''
        check_box_list = request.POST.getlist('check_box_list')
        if check_box_list:
            print(check_box_list)
            money=""
            for i in check_box_list:
                money+=i
			
        else:
            money = 'error'
        #form = LogForm(request.POST)
        #if form.is_valid():
            #print("yes")
        '''
        result = get_ner_result.delay({'txt':txt, 'model':model})#will not wait 
        
        res = result.get(timeout=2)###设为经典值
        time.sleep(5)
        
        ####while
        
        
        whatever = res['restext']
        fp = open('mysite/static/result.html')
        t = Template(fp.read())
        fp.close()
        
        html = t.render(RequestContext(request,{'text' : txt,'model' : model,'restext':whatever}))
    else:
    
        html = t.render(RequestContext(request,{'time':now}))
    return HttpResponse(html)
