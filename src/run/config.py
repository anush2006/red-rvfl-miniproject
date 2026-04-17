# config.py

def get_all_configs():
    configs = []

    '''WINDOW_SIZE_OPTIONS = [48,96,124]
    HIDDEN_SIZE_OPTIONS = [20,50,100,200]
    NUM_LAYER_OPTIONS = [1,3,5,7]
    RIDGE_ALPHA_OPTIONS = [0.001,0.01 ,0.1,1]
    INPUT_SCALING_OPTIONS = [0.1,0.3,0.5,0.7,0.1]
    K_OPTIONS = [1,2,3,4]'''


    WINDOW_SIZE_OPTIONS = [48,]
    HIDDEN_SIZE_OPTIONS = [20,]
    NUM_LAYER_OPTIONS = [1,]
    RIDGE_ALPHA_OPTIONS = [0.01 ,]
    INPUT_SCALING_OPTIONS = [0.7,]
    K_OPTIONS = [1,2,3,4]

    for w in WINDOW_SIZE_OPTIONS:
        for h in HIDDEN_SIZE_OPTIONS:
            for l in NUM_LAYER_OPTIONS:
                for r in RIDGE_ALPHA_OPTIONS:
                    for s in INPUT_SCALING_OPTIONS:
                        for k in K_OPTIONS:
                            configs.append({
                                "window": w,
                                "hidden_size": h,
                                "num_layers": l,
                                "ridge_alpha": r,
                                "input_scaling": s,
                                "k": k
                            })

    return configs