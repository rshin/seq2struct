{
    data: {
        train: {
            name: 'hearthstone', 
            filename: 'data/hearthstone/train_hs'
        },
        val: {
            name: 'hearthstone', 
            filename: 'data/hearthstone/dev_hs'
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'NL2Code',
        },   
        decoder: {
            name: 'NL2Code',
            # dropout
        },
        encoder_preproc: {
            save_path: 'data/hearthstone/nl2code/',
            min_freq: 3,
            max_count: 5000,
        },
        decoder_preproc: self.encoder_preproc,
    },

    train: {
        batch_size: 10,
        keep_every_n: 100,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 1,
        max_steps: 2650,
    },
    optimizer: {
        name: 'adadelta',
        lr: 1.0,
        rho: 0.95,
        eps: 1e-6,
    },

}