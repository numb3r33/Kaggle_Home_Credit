#!/bin/bash
python v131.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
python v131.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231
python v132.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
python v132.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231
python v133.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
python v133.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231
python v134.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
python v134.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231