import act

class ConfigFormatError(Exception):
    """Configuration file format error."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class LayerSpec:
    """A class for the type of activation function of a layer"""
    def __init__(self, out_dim, activation, 
            weight_decay=0, weight_constraint=0, dropout=0):
        self.act_type = activation
        self.out_dim = out_dim
        self.weight_decay = weight_decay
        self.weight_constraint = weight_constraint
        self.dropout = dropout

class OutputSpec:
    """A class for the type of activation of the output layer"""
    def __init__(self, out_dim, output_type, 
            weight_decay=0, weight_constraint=0, dropout=0):
        self.output_type = output_type
        self.out_dim = out_dim
        self.weight_decay = 0
        self.weight_constraint = 0
        self.dropout = 0

class Config:
    """A class for configuration of the neural net."""
    
    # section names
    SEC_DATA       = 'data'
    SEC_PARAMETERS = 'parameters'
    SEC_LAYER      = 'layer'
    SEC_OUTPUT     = 'output'

    # attribute names
    DATA_TRAIN_FILE      = 'train_data_file'
    DATA_VAL_FILE        = 'val_data_file'
    DATA_TEST_FILE       = 'test_data_file'
    DATA_OUTPUT_DIR      = 'output_dir'
    DATA_TRAIN_LOSS_FILE = 'train_loss_file'
    DATA_TASK_LOSS_FILE  = 'task_loss_file'
    DATA_IS_REGRESSION   = 'is_regression'

    PAR_LEARN_RATE       = 'learn_rate'
    PAR_LR_DROP_RATE     = 'lr_drop_rate'
    PAR_INIT_SCALE       = 'init_scale'
    PAR_INIT_MOMENTUM    = 'init_momentum'
    PAR_SWITCH_EPOCH     = 'switch_epoch'
    PAR_FINAL_MOMENTUM   = 'final_momentum'
    PAR_WEIGHT_DECAY     = 'weight_decay'
    PAR_MINIBATCH_SIZE   = 'minibatch_size'
    PAR_NUM_EPOCHS       = 'num_epochs'
    PAR_EPOCH_TO_DISPLAY = 'epoch_to_display'
    PAR_DISPLAY_WINC     = 'display_winc'
    PAR_EPOCH_TO_SAVE    = 'epoch_to_save'
    PAR_SHOW_TASK_LOSS   = 'show_task_loss'
    PAR_SHOW_ACCURACY    = 'show_accuracy'
    PAR_RANDOM_SEED      = 'random_seed'
    PAR_INPUT_NOISE      = 'input_noise'

    LAYER_TYPE              = 'type'
    LAYER_OUT_DIM           = 'out_dim'
    LAYER_WEIGHT_DECAY      = 'weight_decay'
    LAYER_WEIGHT_CONSTRAINT = 'weight_constraint'
    LAYER_DROPOUT           = 'dropout'

    OUTPUT_TYPE    = 'type'
    OUTPUT_OUT_DIM = 'out_dim'
    OUTPUT_WEIGHT_DECAY      = 'weight_decay'
    OUTPUT_WEIGHT_CONSTRAINT = 'weight_constraint'
    OUTPUT_DROPOUT           = 'dropout'


    def __init__(self, file_name):
        """Set default option values. Read configurations from a text file."""
        # layer & output managers
        self.act_manager = act.ActivationTypeManager()
        self.output_manager = act.OutputTypeManager()

        # data files
        self.train_data_file = ''
        self.val_data_file = ''
        self.test_data_file = ''
        self.is_val = False
        self.is_test = False
        self.output_dir = ''
        self.output_filename_pattern = 'm%d'
        self.is_output = False
        self.train_loss_file = None
        self.task_loss_file = None
        self.is_regression = False

        # parameters for training
        self.learn_rate = 0.001
        self.lr_drop_rate = 10
        self.init_scale = 0.001
        self.init_momentum = 0.9
        self.switch_epoch = 0
        self.final_momentum = 0.9
        self.weight_decay = 0
        self.minibatch_size = 100
        self.epoch_to_display = 10
        self.display_winc = True 
        self.epoch_to_save = 100
        self.show_task_loss = False
        self.show_accuracy = False
        self.num_epochs = 1000
        self.random_seed = None
        self.input_noise = 0

        # information for each layer, read from the file
        self.num_layers = 0

        # default output layer
        self.output = OutputSpec(10, self.output_manager.linear)

        self._parse_cfg_file(file_name)

    def _parse_cfg_file(self, file_name):
        """Parse a configuration file."""
        import ConfigParser

        cfg = ConfigParser.ConfigParser()
        cfg.read(file_name)
        # make sure the config file contains all required sections
        self._check_cfg_sections(cfg)
        
        # starts parsing
        self.num_layers = len(cfg.sections()) - 3
        self.layer = []

        # parse data section
        if cfg.has_option(Config.SEC_DATA, Config.DATA_TRAIN_FILE):
            self.train_data_file = cfg.get(Config.SEC_DATA, Config.DATA_TRAIN_FILE)
        else:
            raise ConfigFormatError('No training data.')

        if cfg.has_option(Config.SEC_DATA, Config.DATA_VAL_FILE):
            self.val_data_file = cfg.get(Config.SEC_DATA, Config.DATA_VAL_FILE)
            self.is_val = True

        if cfg.has_option(Config.SEC_DATA, Config.DATA_TEST_FILE):
            self.test_data_file = cfg.get(Config.SEC_DATA, Config.DATA_TEST_FILE)
            self.is_test = True
            
        if cfg.has_option(Config.SEC_DATA, Config.DATA_OUTPUT_DIR):
            self.output_dir = cfg.get(Config.SEC_DATA, Config.DATA_OUTPUT_DIR)
            self.is_output = True

        if cfg.has_option(Config.SEC_DATA, Config.DATA_IS_REGRESSION):
            self.is_regression = (cfg.get(Config.SEC_DATA, 
                Config.DATA_IS_REGRESSION).lower() == 'true')

        if cfg.has_option(Config.SEC_DATA, Config.DATA_TRAIN_LOSS_FILE):
            self.train_loss_file = cfg.get(Config.SEC_DATA, Config.DATA_TRAIN_LOSS_FILE)

        if cfg.has_option(Config.SEC_DATA, Config.DATA_TASK_LOSS_FILE):
            self.task_loss_file = cfg.get(Config.SEC_DATA, Config.DATA_TASK_LOSS_FILE)

        # parse parameter section
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_LEARN_RATE):
            self.learn_rate = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_LEARN_RATE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_LR_DROP_RATE):
            self.lr_drop_rate = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_LR_DROP_RATE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_INIT_SCALE):
            self.init_scale = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_INIT_SCALE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_INIT_MOMENTUM):
            self.init_momentum = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_INIT_MOMENTUM))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_SWITCH_EPOCH):
            self.switch_epoch = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_SWITCH_EPOCH))
        else:   # if no switch point is defined
            self.switch_epoch = 0
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_FINAL_MOMENTUM):
            self.final_momentum = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_FINAL_MOMENTUM))
        else:   # if no final momentum is defined, use initial momentum as the final
            self.final_momentum = self.init_momentum
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_WEIGHT_DECAY):
            self.weight_decay = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_WEIGHT_DECAY))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_MINIBATCH_SIZE):
            self.minibatch_size = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_MINIBATCH_SIZE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_NUM_EPOCHS):
            self.num_epochs = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_NUM_EPOCHS))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_EPOCH_TO_DISPLAY):
            self.epoch_to_display = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_EPOCH_TO_DISPLAY))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_DISPLAY_WINC):
            self.display_winc = (cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_DISPLAY_WINC).lower() == 'true')
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_SHOW_TASK_LOSS):
            self.show_task_loss = (cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_SHOW_TASK_LOSS).lower() == 'true')
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_SHOW_ACCURACY):
            self.show_accuracy = (cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_SHOW_ACCURACY).lower() == 'true')
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_EPOCH_TO_SAVE):
            self.epoch_to_save = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_EPOCH_TO_SAVE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_RANDOM_SEED):
            self.random_seed = int(cfg.get(Config.SEC_PARAMETERS,
                Config.PAR_RANDOM_SEED))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_INPUT_NOISE):
            self.input_noise = float(cfg.get(Config.SEC_PARAMETERS,
                Config.PAR_INPUT_NOISE))

        # parse layer specifications
        for i in range(1, self.num_layers + 1):
            layer_name = Config.SEC_LAYER + str(i)
            if cfg.has_option(layer_name, Config.LAYER_TYPE) and \
                    cfg.has_option(layer_name, Config.LAYER_OUT_DIM):

                new_layer = LayerSpec(
                        int(cfg.get(layer_name, Config.LAYER_OUT_DIM)),
                        self.act_manager.get_type_by_name(
                            cfg.get(layer_name, Config.LAYER_TYPE)))

                if cfg.has_option(layer_name, Config.LAYER_WEIGHT_DECAY):
                    new_layer.weight_decay = float(cfg.get(layer_name, 
                        Config.LAYER_WEIGHT_DECAY))
                if cfg.has_option(layer_name, Config.LAYER_WEIGHT_CONSTRAINT):
                    new_layer.weight_constraint = float(cfg.get(layer_name,
                        Config.LAYER_WEIGHT_CONSTRAINT))
                if cfg.has_option(layer_name, Config.LAYER_DROPOUT):
                    new_layer.dropout = float(cfg.get(layer_name,
                        Config.LAYER_DROPOUT))

                self.layer.append(new_layer)
            else:
                raise ConfigFormatError('Incomplete layer: ' + layer_name)

        # parse output section
        if cfg.has_option(Config.SEC_OUTPUT, Config.OUTPUT_TYPE) and \
                cfg.has_option(Config.SEC_OUTPUT, Config.OUTPUT_OUT_DIM):
            self.output = OutputSpec(
                    int(cfg.get(Config.SEC_OUTPUT, Config.OUTPUT_OUT_DIM)),
                    self.output_manager.get_output_by_name(
                        cfg.get(Config.SEC_OUTPUT, Config.OUTPUT_TYPE)))

            if cfg.has_option(Config.SEC_OUTPUT, Config.OUTPUT_WEIGHT_DECAY):
                self.output.weight_decay = float(cfg.get(Config.SEC_OUTPUT,
                    Config.OUTPUT_WEIGHT_DECAY))
            if cfg.has_option(Config.SEC_OUTPUT, Config.OUTPUT_WEIGHT_CONSTRAINT):
                self.output.weight_constraint = float(cfg.get(Config.SEC_OUTPUT,
                    Config.OUTPUT_WEIGHT_CONSTRAINT))
            if cfg.has_option(Config.SEC_OUTPUT, Config.OUTPUT_DROPOUT):
                self.output.dropout = float(cfg.get(Config.SEC_OUTPUT,
                    Config.OUTPUT_DROPOUT))
        else:
            raise ConfigFormatError('Incomplete output layer.')

    def _check_cfg_sections(self, cfg):
        """The config parser cfg should already have the config file loaded.
        """
        secs = cfg.sections()

        n_data_sec = 0
        n_par_sec = 0
        n_output_sec = 0
        current_layer = 0

        for section in secs:
            if section == Config.SEC_DATA:
                n_data_sec += 1
                if n_data_sec > 1:
                    raise ConfigFormatError('Multiple data sections.')
            elif section == Config.SEC_PARAMETERS:
                n_par_sec += 1
                if n_par_sec > 1:
                    raise ConfigFormatError('Multiple parameter sections.')
            elif section == Config.SEC_OUTPUT:
                n_output_sec += 1
                if n_output_sec > 1:
                    raise ConfigFormatError('Multiple output sections.')
            elif section.startswith(Config.SEC_LAYER):
                n_layer = int(section[5:])
                if not n_layer == current_layer + 1:
                    raise ConfigFormatError('Repeated/skiped layer definition.')
                current_layer += 1
            else:
                raise ConfigFormatError('Unknown section name.')

        return True

    def display(self):
        """For debug use only."""

        # data section
        print '[' + Config.SEC_DATA + ']'
        print Config.DATA_TRAIN_FILE +  '=' + self.train_data_file
        print Config.DATA_VAL_FILE +    '=' + self.val_data_file
        print 'is_val=' + str(self.is_val)
        print Config.DATA_TEST_FILE +   '=' + self.test_data_file
        print 'is_test=' + str(self.is_test)
        print Config.DATA_OUTPUT_DIR +  '=' + self.output_dir
        print 'is_output=' + str(self.is_output)
        print Config.DATA_TRAIN_LOSS_FILE + '=' + str(self.train_loss_file)
        print Config.DATA_TASK_LOSS_FILE  + '=' + str(self.task_loss_file)
        print 'is_regression=' + str(self.is_regression)
        print ''

        # parameter section
        print '[' + Config.SEC_PARAMETERS + ']'
        print Config.PAR_LEARN_RATE +       '=' + str(self.learn_rate)
        print Config.PAR_INIT_SCALE +       '=' + str(self.init_scale)
        print Config.PAR_INIT_MOMENTUM +    '=' + str(self.init_momentum)
        print Config.PAR_SWITCH_EPOCH +     '=' + str(self.switch_epoch)
        print Config.PAR_FINAL_MOMENTUM +   '=' + str(self.final_momentum)
        print Config.PAR_WEIGHT_DECAY +     '=' + str(self.weight_decay)
        print Config.PAR_MINIBATCH_SIZE +   '=' + str(self.minibatch_size)
        print Config.PAR_NUM_EPOCHS +       '=' + str(self.num_epochs)
        print Config.PAR_EPOCH_TO_DISPLAY + '=' + str(self.epoch_to_display)
        print Config.PAR_DISPLAY_WINC +     '=' + str(self.display_winc)
        print Config.PAR_EPOCH_TO_SAVE +    '=' + str(self.epoch_to_save)
        print ''

        # layer sections
        for i in range(0, self.num_layers):
            print '[' + Config.SEC_LAYER + str(i+1) + ']'
            print Config.LAYER_TYPE +    '=' + str(self.layer[i].act_type)
            print Config.LAYER_OUT_DIM + '=' + str(self.layer[i].out_dim)
            print ''

        # output section
        print '[' + Config.SEC_OUTPUT + ']'
        print Config.OUTPUT_TYPE +    '=' + str(self.output.output_type)
        print Config.OUTPUT_OUT_DIM + '=' + str(self.output.out_dim)

