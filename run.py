from flask import Flask, render_template, request
from werkzeug import secure_filename
import os ,cv2
import collections as cl
import json
#from inference_usbCam_face import Main
from updata_face_and_bubble import Main

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
UPLOAD_FOLDER = './img'
image_dir="img"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST'])
def test():
    if request.method == 'POST':
        img_file=request.files["post_data"]
        if img_file:
            filename = secure_filename(img_file.filename)
            print(filename)  #確認
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path="./img/"+filename

            faces_list,rect_list=Main(image_path)

            data=cl.OrderedDict()

            data["num"]=len(faces_list)
            for i,face in enumerate(faces_list):
                data["x_position"+str(i)]=int(face[0])
                data["y_position"+str(i)]=int(face[1])
                data["x_length"+str(i)]=int(face[2])
                data["y_length"+str(i)]=int(face[3])
                data["center_position_x"+str(i)]=int(face[0])+int(int(face[2])/2)
                data["center_position_y"+str(i)]=int(face[1])+int(int(face[3])/2)

            data["rect_num"]=len(rect_list)
            for i,rect in enumerate(rect_list):
                data["rect_x_position"+str(i)]=int(rect[0])
                data["rect_y_position"+str(i)]=int(rect[1])
                data["rect_x_length"+str(i)]=int(rect[2])
                data["rect_y_length"+str(i)]=int(rect[3])
                data["rect_center_position_x"+str(i)]=int(rect[0])+int(int(rect[2])/2)
                data["rect_center_position_y"+str(i)]=int(rect[1])+int(int(rect[3])/2)
            print("{}".format(json.dumps(data,indent=4)))
        return json.dumps(data,indent=4)

    #return render_template("index.html",s=10)


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)
