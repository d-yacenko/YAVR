# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from tempfile import SpooledTemporaryFile
from cgi import FieldStorage
import sys, os
from PIL import Image
import  segmentation 
import threading
import time

#####################################################################
################## server starting ##################################
#####################################################################

hostName = "192.168.49.110"
serverPort = 8080
#temp_file= "/run/user/"+str(os.getuid())+"/live_image.jpg"
temp_file= "/RAM/live_image.jpg"
run_thread= None

import matplotlib.pyplot as plt
import io


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is a body segmentation web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))


    def do_POST(self):
        start=time.time()
        global run_thread
        length = int(self.headers['Content-Length'])
        #print(self.headers.get('content-type', '').lower())
        with  io.BytesIO() as imgfile:
            read = 0
            while read < length:
                if length-read > 1024:
                    buffer = self.rfile.read(1024)
                else:
                    buffer = self.rfile.read(length-read)
                imgfile.write(buffer)
                read += len(buffer)
            imgfile.seek(0)
            #print(read)
            #print('loaded')
            imgfile.seek(100)
            ff=open(temp_file,'wb')
            ff.write(imgfile.read(read-100))
            ff.close()
            if run_thread is None:
                run_thread = threading.Thread(target=segmentation.process, args=(temp_file,))
                run_thread.start()
            else:    
                if not run_thread.is_alive():
                    run_thread = threading.Thread(target=segmentation.process, args=(temp_file,))
                    run_thread.start()
                else:
                    pass
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        head, tail = os.path.split(temp_file)
        with open(head+"/out"+tail,'rb') as fo:
            fsize = os.path.getsize(head+"/out"+tail) 
            #print(fsize)
            response.write(fo.read(fsize))
        self.wfile.write(response.getvalue())
        end=time.time()
        #print(f"Runtime of the program is {end - start}")


if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

