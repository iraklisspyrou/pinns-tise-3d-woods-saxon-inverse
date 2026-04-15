####################################
#                                  #
#    TISE PINN INVERSE PROBLEM     #
#                                  #
####################################

# import necessary libs here


# define model netwrork architectures

# MLP data part ( I/O: wavefunctions -> energy states)
# define # of layers
# in_dims = 3 (r = x,y,z) (T.B.D)
# hidd_dims = N
# out_dims = 1 (T.B.D SEE THE INPUT)
# activation function = SIREN (depending on the input)


# MLP PHYSICS (WS parameters) (I/O: energy (T.B.D) -> N parameters of WS
# define N layers
# in_dims = 1 (E)
# hidd_dims = N
# out_dims = N
# activation function (




# define optimizer
# define loss (for now MSE)



############## Training loop #################

# optimizer_zero grad

# models

# calculate L = αLdata + βLphys

# Loss backward

# optimizer step



################ Evaluation #################33

# plot parameter maps
# print energy states
# create GT parameter maps and calculate NMSE


