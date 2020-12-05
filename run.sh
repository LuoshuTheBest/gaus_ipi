#Set up MD parameters
#Units
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
export total_steps=3
export time_step=0.5
#For NVT and NPT
export tau=1
export therm_mode='langevin'
#For NPT
export pressure=10
export baro_mode='isotropic'

#Option of output: prop(properties), pos(trajectory), force, chk(checkpoint)
export output_opt="prop pos vel chk"
#####################################################################################

source /home/luoshu/i-pi/env.sh
export mol=o2h.xyz
#moles=test; sbatch bash_opt_df.slurm $moles
bash clear.sh
python3 gen_input.py $mol
i-pi input.xml & 
python3 gauss_driver.py $mol

