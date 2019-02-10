from train import Trainer
import time

# Command Line Argument Method
HEIGHT = 28
WIDTH = 28
CHANNEL = 1
LATENT_SPACE_SIZE = 100
EPOCHS = 40000
BATCH = 32
CHECKPOINT = 500
MODEL_TYPE = -1

start = time.time()
trainer = Trainer(height=HEIGHT,
                  width=WIDTH,
                  channels=CHANNEL,
                  latent_size=LATENT_SPACE_SIZE,
                  epochs=EPOCHS,
                  batch=BATCH,
                  checkpoint=CHECKPOINT,
                  model_type=MODEL_TYPE)

trainer.train()
end = time.time()
print('Elaplsed Time: %.2f minutes' % ((end-start)/60))