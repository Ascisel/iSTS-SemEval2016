#!/bin/bash

source setup_environment.sh
python prepare_data.py --datasets=headlines,images,answers-students
python main.py --model=separate
python main.py --model=connected

mv separate_results.xml separate_results.wa
mv connected_results.xml connected_results.wa

./evalF1_no_penalty.pl test/STSint.testinput.headlines.wa separate_results.wa
./evalF1_no_penalty.pl test/STSint.testinput.headlines.wa connected_results.wa
