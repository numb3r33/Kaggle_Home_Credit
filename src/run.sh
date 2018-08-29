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

# Stacker ( 1st-stage )
# python v146.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v146.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -t True -cv_seed 4457 -seed 1235

# LGB (dart)
# python v147.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v147.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231

# Stacker ( 2nd-stage )
# python v148.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231
# python v149.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231


# LGB ( gain_score > .3)
# python v150.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v150.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231


# LGB ( cv_adversarial_idx_v1.csv, gain_score > .3 )
python v151.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231


# Stacker ( 1st-stage )
# python v152.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
# python v152.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -t True -cv_seed 4457 -seed 1235

# LGB ( XGBOOST Leaves )
# python v153.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv_predict True -cv_seed 4457 -seed 1231


# Stacker ( 1st-stage )
python v154.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -cv True -seed 4457
python v154.py -input_path data/raw/ -output_path data/interim/ -data_folder dataset6/ -t True -cv_seed 4457 -seed 1235
