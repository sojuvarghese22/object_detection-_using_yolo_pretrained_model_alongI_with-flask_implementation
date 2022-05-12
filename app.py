from flask import Flask,jsonify,request

from yolo_detection_images import detectObjects
app = Flask(__name__)

@app.route('/myapp/detectObjects')
def detect():
    img = request.args['image']
    print(img)
    img_path= 'images/'+img
    r =detectObjects(img_path)
    return jsonify(r)



app.run()