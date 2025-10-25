## Utils.py -- Some utility functions
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Chun-Chen Tu <timtu@umich.edu>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.from PIL import Image

from PIL import Image
import os, numpy as np

def save_img(img, name="output.png"):
    if os.path.dirname(name):
        os.makedirs(os.path.dirname(name), exist_ok=True)
    np.save(name, img)
    fig = np.around((img + 0.5) * 255.0)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    pic.save(name)

def generate_attack_data_set_specific(data, model, MGR, target_label, nFunc):
    """Generate adversarial dataset for specific target labels (interface unchanged)"""
    images, labels = data.test_data, data.test_labels
    target_indices = [i for i in range(len(labels)) if np.argmax(labels[i]) == target_label]
    sel = np.random.choice(target_indices, min(nFunc, len(target_indices)), replace=False)
    origImgs = images[sel].astype(np.float32)
    origLabels = labels[sel].astype(np.float32)
    print(f"Selected {len(sel)} images for target digit {target_label}")
    return origImgs, origLabels, sel