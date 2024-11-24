conda activate sbi_emri 

# set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create a dynamic output filename
output_filename="output.txt"

# Execute the Python command and redirect output to the dynamic filename
nohup python ../mcmc_code/mcmc_run.py > $output_filename &
