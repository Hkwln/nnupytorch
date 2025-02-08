#!/bin/bash
current_dir ="$(cd "$dirname "${BASH_SOURCE[0]}")" && pwd)"
alias sentiment=  current_dir

echo hi there, here you can run the python script
python "$current_dir/interactive_predictions.py"

