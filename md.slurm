#!/bin/bash
# parallel job using 128 cores. and runs for 4 hours (max)
#SBATCH -N 1 # node count
#SBATCH -t 48:00:00
# sends mail when process begins, and 
# when it ends. Make sure you define your email 
# address.
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
##SBATCH --mail-user=juny@hku.hk

#make sure calculations are carried out in /scratch on each node
export time_unit=femtosecond
export temp_unit=kelvin
export potential_unit=electronvolt
export press_unit=bar
#megapascal
export cell_units=angstrom
export force_unit=piconewton

####################################################################################
#Parameters
export verbose=5
export charge=1
export dyn_mode='nvt'
export stride='1'
export nbeads='1'
export port=51928
export seed=3348
export temperature=300
export total_steps=2
export time_step=0.5
#For NVT and NPT
export tau=1
export therm_mode='langevin'
#For NPT
export pressure=10
export baro_mode='isotropic'
export output_opt="prop pos vel chk"
#####################################################################################

source /home/dhzhang/software/i-pi-master/env.sh
export mol=o2h.xyz
#moles=test; sbatch bash_opt_df.slurm $moles
bash clear.sh
python gen_input.py $mol
i-pi input.xml &
python gauss_driver.py $mol

