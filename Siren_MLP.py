
# ============================
#  Input (t) -> Caif
# ============================

class MLP_aif(nn.Module):
  def __init__(self,
                 num_layers_aif,
                 dim_hidden_aif,
                 dim_in=3,
                 dim_out=1,
                 w0_aif = 10., w0_initial_aif = 10., use_bias = True,
                 final_activation = None):
        super(MLP_aif, self).__init__()
        self.n_layers = num_layers_aif
        self.n_units = dim_hidden_aif
        self.n_inputs = dim_in
        self.neurons_out = dim_out
        self.w0 = w0_aif
        self.w0_initial = w0_initial_aif
        self.use_bias = use_bias
        self.final_activation = final_activation
        self.net = self.__make_net()

  def __make_net(self):
    layers = []
    for i in range(self.n_layers):
      is_first = i == 0
      layer_w0 = self.w0_initial if is_first else self.w0
      layer_dim_in = self.n_inputs if is_first else self.n_units
      layers.append(Siren(
          dim_in=layer_dim_in,
          dim_out=self.n_units,
          w0=layer_w0,
          use_bias=self.use_bias,
          is_first=is_first
      ))

    final_activation = nn.Identity() if not exists(self.final_activation) else self.final_activation
    layers.append(Siren(
          dim_in = self.n_units,
          dim_out = self.neurons_out,
          w0 = self.w0,
          use_bias = self.use_bias,
          activation = final_activation))
    return nn.Sequential(*layers)

  def forward(self, t):
    return self.net(t)

def exists(val):
    return val is not None


class MLP_myo(nn.Module):
  def __init__(self,
                 num_layers_myo,
                 dim_hidden_myo,
                 dim_in=3,
                 dim_out=1,
                 w0_myo = 10., w0_initial_myo = 10., use_bias = True,
                 final_activation = None):
        super(MLP_myo, self).__init__()
        self.n_layers = num_layers_myo
        self.n_units = dim_hidden_myo
        self.n_inputs = dim_in
        self.neurons_out = dim_out
        self.w0 = w0_myo
        self.w0_initial = w0_initial_myo
        self.use_bias = use_bias
        self.final_activation = final_activation
        self.net = self.__make_net()

  def __make_net(self):
    layers = []
    for i in range(self.n_layers):
      is_first = i == 0
      layer_w0 = self.w0_initial if is_first else self.w0
      layer_dim_in = self.n_inputs if is_first else self.n_units
      layers.append(Siren(
          dim_in=layer_dim_in,
          dim_out=self.n_units,
          w0=layer_w0,
          use_bias=self.use_bias,
          is_first=is_first
      ))

    final_activation = nn.Identity() if not exists(self.final_activation) else self.final_activation
    layers.append(Siren(
          dim_in = self.n_units,
          dim_out = self.neurons_out,
          w0 = self.w0,
          use_bias = self.use_bias,
          activation = final_activation))
    return nn.Sequential(*layers)

  def forward(self, t, xy):
    txy = torch.concat([t, xy], dim=-1)

    return self.net(txy)


class MLP_ode_delay(nn.Module):
    def __init__(self,
                 num_layers_2cxm,
                 dim_hidden_2cxm,
                 dim_in=2,
                 w0_2cxm=10., w0_initial_2cxm=10., use_bias=True):
        super(MLP_ode_delay, self).__init__()
        self.n_layers = num_layers_2cxm
        self.n_units = dim_hidden_2cxm
        self.n_inputs = dim_in
        self.w0 = w0_2cxm
        self.w0_initial = w0_initial_2cxm
        self.use_bias = use_bias

        # --- trunk (shared SIREN layers) ---
        layers = []
        for i in range(self.n_layers):
            is_first = i == 0
            layer_w0 = self.w0_initial if is_first else self.w0
            layer_dim_in = self.n_inputs if is_first else self.n_units
            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=self.n_units,
                w0=layer_w0,
                use_bias=self.use_bias,
                is_first=is_first
            ))
        self.trunk = nn.Sequential(*layers)

        # --- physiology head (4 outputs) ---
        self.head_phys = Siren(
            dim_in=self.n_units,
            dim_out=4,
            w0=self.w0,
            use_bias=self.use_bias,
            activation=nn.Softplus()
        )

        # --- delay head (1 output) ---
        self.head_delay = Siren(
            dim_in=self.n_units,
            dim_out=1,
            w0=self.w0,
            use_bias=self.use_bias,
            activation=nn.Softplus() 
        )

    def forward(self, xy):
        shared = self.trunk(xy)             
        phys = self.head_phys(shared)       
        delay = self.head_delay(shared)    
        return phys, delay

        
class MLP_ode(nn.Module):
  def __init__(self,
                 num_layers_2cxm,
                 dim_hidden_2cxm,
                 dim_in=2,
                 dim_out=4,
                 w0_2cxm = 10., w0_initial_2cxm = 10., use_bias = True,
                 final_activation = None):
        super(MLP_ode, self).__init__()
        self.n_layers = num_layers_2cxm
        self.n_units = dim_hidden_2cxm
        self.n_inputs = dim_in
        self.neurons_out = dim_out
        self.w0 = w0_2cxm
        self.w0_initial = w0_initial_2cxm
        self.use_bias = use_bias
        self.final_activation = final_activation
        self.net = self.__make_net()

  def __make_net(self):
    layers = []
    for i in range(self.n_layers):
      is_first = i == 0
      layer_w0 = self.w0_initial if is_first else self.w0
      layer_dim_in = self.n_inputs if is_first else self.n_units
      layers.append(Siren(
          dim_in=layer_dim_in,
          dim_out=self.n_units,
          w0=layer_w0,
          use_bias=self.use_bias,
          is_first=is_first
      ))

    final_activation = nn.Identity() if not exists(self.final_activation) else self.final_activation
    layers.append(Siren(
          dim_in = self.n_units,
          dim_out = self.neurons_out,
          w0 = self.w0,
          use_bias = self.use_bias,
          activation = final_activation))
    return nn.Sequential(*layers)

  def forward(self, xy):
    return self.net(xy)  
