#!/bin/bash

# CatBoost
# python v120.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v120.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231

# XGBoost
# python v143.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v143.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231

# NN
# python v140.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v140.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231

# KNN
# python v142.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v142.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231

# LGB ( null importance )
# python v145.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v145.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231

# Ensemble
# python v146.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v146.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -t True -cv_seed 4457 -seed 1235

# LGB (dart)
# python v147.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v147.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231

# 2nd-stage Stacking
python v148.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231
python v149.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231
