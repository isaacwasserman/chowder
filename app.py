from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io
import base64
from chowder import *
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) 
def index():
    if request.method == 'POST': 
        start_time = time.time()
        # Handle form submit
        file = request.files['image']
        # Get form data
        params = request.form
        regions_per_frame = int(params.get('regions_per_frame', 1))
        number_of_frames = int(params.get('number_of_frames', 5))
        regions_considered = int(params.get('regions_considered', 10))
        frames_per_second = int(params.get('frames_per_second', 3))
        task = params.get('task', 'scramble')

        # Read image as blob    
        img_blob = file.read()
        
        # Optional - convert to numpy array
        file_bytes = np.asarray(bytearray(img_blob), dtype=np.uint8)
        img = Image.open(io.BytesIO(file_bytes))
        img_array = np.array(img) 
        
        # Process image
        masks = segment(img)
        mask_areas = [mask.sum() for mask in masks]
        masks = np.array(masks)[np.argsort(mask_areas)[::-1]]
        num_regions = regions_considered

        if task == 'scramble':
            frames = selective_scramble(img_array, masks[:num_regions], num_frames=number_of_frames, regions_per_frame=regions_per_frame)
        else:
            frames = selective_reveal(img_array, masks[:num_regions], num_frames=number_of_frames, regions_per_frame=regions_per_frame)
        
        gif_blob = frames_to_gif(frames, fps=frames_per_second)

        print(f"Completed request in {time.time() - start_time} seconds")
        
        return render_template('image.html', img_data=gif_blob)

    # On GET show upload form 
    return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True)