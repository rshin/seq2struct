{
    data: {
        train: {
            name: 'hearthstone', 
            filename: 'data/hearthstone/train_hs',
            limit: 2,
        },
        val: {
            name: 'hearthstone', 
            filename: 'data/hearthstone/train_hs',
            limit: 2,
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'NL2Code',
            dropout: 0.2,
        },   
        decoder: {
            name: 'NL2Code',
            dropout: 0.2,
        },
        encoder_preproc: {
            save_path: 'data/hearthstone/nl2code-tiny/',
            min_freq: 1,
            max_count: 5000,
        },
        decoder_preproc: self.encoder_preproc,
    },

    train: {
        batch_size: 2,
        eval_batch_size: self.batch_size,

        keep_every_n: 10,
        eval_every_n: 10,
        save_every_n: 10,
        report_every_n: 2,

        max_steps: 2650,
        num_eval_items: 2,
    },
    optimizer: {
        name: 'adadelta',
        lr: 1.0,
        rho: 0.95,
        eps: 1e-6,
    },

}
