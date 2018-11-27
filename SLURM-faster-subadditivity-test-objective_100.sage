#! /home/wangjw/sage/sage-8.3/sage
#SBATCH --array=0-3314 --time 3:00:00
#
# --array: Specify the range of the array tasks.
# --time: Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds" etc.
import os
task_id = os.getenv("SLURM_ARRAY_TASK_ID")
if task_id:
    task_id = int(task_id)

print("Sage hello from task {}".format(task_id))

import sys
sys.path = [''] + sys.path

import igp
from igp import *

logging.disable(logging.INFO)

readfile_path='./test_functions_csv_library/'
allFiles=glob.glob(os.path.join('./test_functions_csv_library/*.csv'))
readfile_name=allFiles[task_id][len(readfile_path):]
writefile_path='./result_objective_100/'

write_performance_file_objective(readfile_path,readfile_name,writefile_path,epsilon=-QQ(1)/100)



