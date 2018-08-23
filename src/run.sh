#!/bin/bash
python v127.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1234
python v128.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1234