## At this point in time...
At time of writing, I comfortably can obtain an ECG from the 
downloaded PTB-XL database via the .csv file that comes with it and the python
library pandas

The shape of a ECG (specifically sampled at 500 Hz) is (5000, 12). 
- 5000 comes from 500 Hz x 10 seconds = 5,000 recordings
- 12 comes from 12 leads

Now I want to pass the ECG into an autoencoder to try extract features

**Eventual goal is to use a Beta VAE**

For now I will try to build a simple autoencoder capable of taking in an ECG, 
compressing it and reconstructing it

