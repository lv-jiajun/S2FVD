#!/usr/bin/python3

from xml.dom.minidom import parse
import xml.dom.minidom

# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse("C:\\Users\\Administrator\\Desktop\\astnn-master\\experience\\test.xml")
DOMTree = xml.dom.minidom.parse(
    "C:\\Users\\Administrator\\Desktop\\Model\\SySeVR-master\\Program data\\SARD\\SARD.xml")
collection = DOMTree.documentElement

# 在集合中获取所有电影
movies = collection.getElementsByTagName("testcase")

# 打印每部电影的详细信息
for movie in movies:
    # type = movie.getElementsByTagName('description')[0]
    # print ("description: %s" % type.childNodes[0].data)

    format = movie.getElementsByTagName('file')[0]
    if format.hasAttribute("path"):
        print(format.getAttribute("path"))
    flaw = movie.getElementsByTagName('flaw')[0]
    if flaw.hasAttribute("name"):
        print(flaw.getAttribute("name"))
