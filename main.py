from cat_dog import CatDogCNN, DatasetBuilder, ModelTrainer, TrainingConfig
from setup_gpu import TensorFlowConfig


def main() -> None:
    TensorFlowConfig.init_gpu()

    config = TrainingConfig()
    train_ds, val_ds = DatasetBuilder(config).build_train_and_val()

    model = CatDogCNN()
    # model.build(input_shape=(None, config.image_size[0], config.image_size[1], 3))
    model.summary()

    trainer = ModelTrainer(model=model, config=config)
    trainer.train(train_ds=train_ds, val_ds=val_ds)


if __name__ == "__main__":
    main()