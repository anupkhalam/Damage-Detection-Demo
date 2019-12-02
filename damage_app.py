# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:28:27 2019

@author: anup
"""

from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import torch
import network_01 as nw
from utils_01 import *
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL = nw.multi_label_vgg_model(training=True)
MODEL.load_state_dict(torch.load('model/model_single_v7.pt', map_location='cpu'))
MODEL = MODEL.to(device)

@app.route('/', methods=['GET','POST'])
def home_page():
    return render_template('base_client.html')


@app.route('/damagedet', methods=['POST'])
def damage_det():
    global MODEL
    if request.method == 'POST':
        loc_image = request.files['image']
#        result = test_image_api(MODEL, [loc_image], device)
        result_1 = test_image_api_multi(MODEL, [loc_image], device)
        print(result_1)
        result_1 = [(k[0].upper(), 'DAMAGE' if k[1]== 'dam' else 'NO DAMAGE') for k in result_1]
        result_disp = {}
        result_disp['cat'] = result[0].upper()
        if result[1] == 'dam':
            result_disp['stat'] = 'DAMAGE'
            output = jsonify(result=result_disp)
            return output

        if result[1] == 'nodam':
            result_disp['stat'] = 'NO DAMAGE'
            output = jsonify(result=result_disp)
            return output
        return jsonify({result:[{'cat':'NA','stat':'NA'}]})

@app.route('/multidamagedet', methods=['POST'])
def damage_det_multi():
    global MODEL
    if request.method == 'POST':
        loc_image = request.files['image']
        result = test_mlable_api(MODEL, [loc_image], device)
        result = [(k[0], 'DAMAGE' if k[1]== 'DAM' else 'NO DAMAGE') for k in result]
        result_list = []
        if len(result) == 0:
            return jsonify({result:[{'cat':'NA','stat':'NA'}]})
        for idx, r in enumerate(result):
            result_disp = {}
            result_disp['cat'] = result[idx][0]
            result_disp['stat'] = result[idx][1]
            result_list.append(result_disp)
        output = jsonify(result=result_list)
        return output
        #return jsonify({result:[{'cat':'NA','stat':'NA'}]})
            #result_disp['cat'] = result[0][0]
'''
	    if result[0][1] == 'dam':
	       result_disp['stat'] = 'DAMAGE'
	       output = jsonify(result=result_disp)
	       return output

	    if result[1] == 'nodam':
	       result_disp['stat'] = 'NO DAMAGE'
	       output = jsonify(result=result_disp)
	       return output
'''


@app.route('/multiimgdamagedet', methods=['POST'])
def damage_det_multi_img():
    global MODEL
    if request.method == 'POST':
        final_result = {}
        for index, img in enumerate(request.files.getlist('file')):
            result = test_mlable_api(MODEL, [img], device)
            result = [(k[0], 'PERIL DAMAGE' if k[1]== 'PERIL' else 'NO DAMAGE') for k in result]
            result_list = []
            if len(result) == 0:
                return jsonify({result:[{'cat':'NA','stat':'NA'}]})
            for idx, r in enumerate(result):
                result_disp = {}
                result_disp['cat'] = result[idx][0]
                result_disp['stat'] = result[idx][1]
                result_list.append(result_disp)
            final_result[img.filename.split('.')[0].replace(' ', '_')] = result_list
        output = jsonify(result=final_result)
        return output
            #return jsonify({result:[{'cat':'NA','stat':'NA'}]})
                #result_disp['cat'] = result[0][0]


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=12000)



