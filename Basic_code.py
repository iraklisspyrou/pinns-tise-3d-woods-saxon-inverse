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



#################################### HPO #################################################################3

# create a sweep in WandB. This handles all HPO stuff. 
# You set:
# 1) the values you need to check
# 2) method of HPO (grid, bayes etc). Bayes works good especially for more detailed HPO
# 3) you set the target to minimize, is that total loss?, specific model parameters? a metric?
# 4) Count = how many runs you want to do (total estimation of the HPO tuning is counts*time for a single training.
# 5) you can also logg the time for each trianing to see if you can get a good hyperparameter combination with less time

def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        t = time.time()

        # Load data
        aif_time, aif, myo_curves, myo_pars = make_dro(-1, 0, 17.5) # type: ignore

        # Call training using config values
        model_aif, model_myo, model_ode, total_losses = train_pinn(myo_curves, myo_pars, aif_time, aif, 
               epochs=3000, lr=1e-4, batch_size=256,
               num_layers_aif=5, dim_hidden_aif=config.dim_hidden_aif, w0_aif=25, w0_initial_aif=15,
               num_layers_myo=5,dim_hidden_myo = config.dim_hidden_myo, w0_myo=20, w0_initial_myo=25,
               num_layers_2cxm=4, dim_hidden_2cxm=config.dim_hidden_2cxm ,w0_2cxm=10, w0_initial_2cxm=20,
               loss_w_aif = 5.0, loss_w_myo = 10.0, loss_w_2cxm = 0.01)

        if total_losses and not math.isnan(total_losses[-1]):
            wandb.log({"final_total_loss": total_losses[-1],
                      "total training time":time.time()-t}
            )
        else:
            print(" total_losses is empty or final loss is NaN — training might have failed.")
            wandb.log({"final_total_loss": None})  # or skip it entirely

sweep_config = {
    "method": "bayes",  
    "metric": {
        "name": "final_total_loss",
        "goal": "minimize"
    },
    "parameters": {
        
        "dim_hidden_aif": {"values":[64,128,256,512]},
        "dim_hidden_myo": {"values":[64,128,256,512]},
        "dim_hidden_2cxm":  {"values":[64,128,256,512]},
    }
}



if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep=sweep_config,entity="q-cardia", project="qp_gpt" )
    wandb.agent(sweep_id, function=sweep_train, count=10)
